import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Bot, Send, Paperclip, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { api } from "@/services/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const TypingIndicator = () => (
  <div className="flex items-start gap-3 max-w-[80%]">
    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
      <Bot className="w-4 h-4 text-primary" />
    </div>
    <div className="bg-card border border-border rounded-2xl rounded-tl-md px-4 py-3 shadow-sm">
      <div className="flex gap-1.5">
        <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce [animation-delay:0ms]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce [animation-delay:150ms]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce [animation-delay:300ms]" />
      </div>
    </div>
  </div>
);

const RichContent = ({ content }: { content: string }) => (
  <div className="text-sm text-muted-foreground">
    <ReactMarkdown
      components={{
        h1: ({ children }) => <h1 className="mb-3 text-lg font-semibold text-foreground">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-3 text-base font-semibold text-foreground">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-2 text-sm font-semibold text-foreground">{children}</h3>,
        p: ({ children }) => <p className="mb-3 leading-7 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-3 list-disc space-y-1.5 pl-5 marker:text-primary">{children}</ul>,
        ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1.5 pl-5 marker:text-primary">{children}</ol>,
        li: ({ children }) => <li className="leading-7">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
        em: ({ children }) => <em className="italic text-foreground/90">{children}</em>,
        hr: () => <hr className="my-4 border-border" />,
        code: ({ inline, children }) =>
          inline ? (
            <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[0.9em] text-foreground">{children}</code>
          ) : (
            <code className="block overflow-x-auto rounded-xl bg-secondary p-3 font-mono text-xs text-foreground">{children}</code>
          ),
        pre: ({ children }) => <pre className="mb-3">{children}</pre>,
        blockquote: ({ children }) => (
          <blockquote className="mb-3 border-l-2 border-primary/40 pl-4 italic text-foreground/80">{children}</blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  </div>
);

const AIAdvisor = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const suggestedPromptsQuery = useQuery({
    queryKey: ["advisor-prompts"],
    queryFn: api.advisor.getSuggestedPrompts,
    retry: 1,
    staleTime: 60_000,
  });

  const chatMutation = useMutation({
    mutationFn: async ({ message, history }: { message: string; history: Message[] }) => {
      const response = await api.advisor.chat(
        message,
        history.map(({ role, content }) => ({ role, content })),
      );
      return response;
    },
    onSuccess: (response) => {
      setMessages((prev) => [
        ...prev,
        { id: `assistant-${Date.now()}`, role: "assistant", content: response.content },
      ]);
    },
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatMutation.isPending]);

  const sendMessage = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || chatMutation.isPending) return;

    const history = [...messages];
    const userMsg: Message = { id: `user-${Date.now()}`, role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    chatMutation.mutate({ message: trimmed, history });
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const handleTextareaInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };

  const suggestedPrompts = suggestedPromptsQuery.data ?? [
    "What's the current portfolio VaR under the stressed regime?",
    "Simulate a 2008 liquidity crisis scenario",
    "Explain the top causal links affecting XLF",
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-3.5rem)]">
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-8">
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
                <Sparkles className="w-7 h-7 text-primary" />
              </div>
              <h1 className="text-2xl font-semibold text-foreground tracking-tight">
                CausalStress AI Assistant
              </h1>
              <p className="text-muted-foreground text-sm text-center max-w-md">
                Ask about portfolio risk, regimes, scenarios, or current market conditions.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-2xl w-full">
              {suggestedPrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => sendMessage(prompt)}
                  className="bg-card text-left px-4 py-3 rounded-xl text-sm text-foreground border border-border hover:border-primary/50 hover:shadow-md transition-all duration-200"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto flex flex-col gap-5">
            {messages.map((msg) =>
              msg.role === "user" ? (
                <div key={msg.id} className="flex justify-end">
                  <div className="max-w-[80%] bg-primary text-primary-foreground rounded-2xl rounded-tr-md px-4 py-3 text-sm shadow-sm">
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div key={msg.id} className="flex items-start gap-3 max-w-[85%]">
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                    <Bot className="w-4 h-4 text-primary" />
                  </div>
                  <div className="bg-card border border-border rounded-2xl rounded-tl-md px-4 py-3 shadow-sm">
                    <RichContent content={msg.content} />
                  </div>
                </div>
              ),
            )}
            {chatMutation.isPending && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="border-t border-border glass px-4 py-3 md:px-8">
        <div className="max-w-3xl mx-auto flex items-end gap-2">
          <button
            className="w-9 h-9 rounded-xl border border-border bg-secondary/50 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors shrink-0 mb-0.5"
            title="Attach data"
          >
            <Paperclip className="w-4 h-4" />
          </button>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => { setInput(e.target.value); handleTextareaInput(); }}
            onKeyDown={handleKeyDown}
            placeholder="Ask about portfolio risk, regimes, scenarios..."
            rows={1}
            className="flex-1 resize-none bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 transition-shadow"
            style={{ maxHeight: 160 }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || chatMutation.isPending}
            className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0 mb-0.5 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: "var(--btn-gradient)" }}
            title="Send message"
          >
            <Send className="w-4 h-4 text-primary-foreground" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIAdvisor;
