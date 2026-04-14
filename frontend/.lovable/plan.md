

# Add AI Financial Advisor Chat Page

## Overview
Create a new `/ai-advisor` page with a full chat interface that integrates into the existing layout, uses semantic theme tokens, and includes mock financial data responses.

## Changes

### 1. Create `src/pages/AIAdvisor.tsx`
The main chat page component containing:

- **State**: messages array, input text, isTyping boolean
- **Empty state**: Centered welcome with "CausalStress AI Assistant" heading and 4 suggested prompt pills styled as `glass-hover` cards (e.g., "Analyze portfolio VaR under Stressed regime", "Explain top causal links for XLF", "Simulate 2008 crisis scenario", "Compare regime transition probabilities")
- **Message list**: `flex-1 overflow-y-auto` scrollable area
  - AI messages: left-aligned, `bg-card border border-border` with warm shadow, `Bot` icon avatar, markdown-rendered content
  - User messages: right-aligned, primary terracotta background (`bg-primary text-primary-foreground`)
  - Typing indicator: 3 animated dots in a `bg-card` bubble
- **Mock data**: Pre-seed one exchange showing a user asking about portfolio VaR and an AI response containing a small HTML table with metrics (e.g., "-$65,810 VaR" in destructive red) to demonstrate financial data rendering
- **Input area**: Bottom-fixed `glass` container with:
  - Paperclip/chart icon button on the left (muted, decorative)
  - `<textarea>` with auto-resize, `bg-background border-border`
  - Send button on the right using `btn-gradient-blue` (maps to terracotta in light mode via `--btn-gradient`)
  - Enter to send, Shift+Enter for newline
- **Full height layout**: The page uses `h-[calc(100vh-3.5rem)]` with `flex flex-col` so the message area fills available space and input stays at the bottom

All colors use semantic tokens: `text-foreground`, `text-muted-foreground`, `bg-background`, `bg-card`, `bg-secondary`, `border-border`, `bg-primary`, `text-primary-foreground`.

### 2. Add sidebar nav item — `src/components/layout/Sidebar.tsx`
Add a new nav entry `{ title: "AI Advisor", url: "/ai-advisor", icon: MessageSquare }` (from lucide-react) between Stress Test and the bottom Settings section.

### 3. Add route — `src/App.tsx`
Import `AIAdvisor` and add `<Route path="/ai-advisor" element={<AIAdvisor />} />`.

### 4. Install `react-markdown`
For rendering AI responses with markdown formatting (tables, bold, code blocks).

## Files

| File | Action |
|------|--------|
| `src/pages/AIAdvisor.tsx` | Create — full chat interface |
| `src/components/layout/Sidebar.tsx` | Edit — add MessageSquare nav item |
| `src/App.tsx` | Edit — add route |
| `package.json` | Edit — add `react-markdown` dependency |

