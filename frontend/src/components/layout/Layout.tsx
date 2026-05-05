import { Bell, User, Sun, Moon } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useTheme } from "@/contexts/ThemeContext";
import { checkApiHealth } from "@/services/api";
import Sidebar from "./Sidebar";

const Layout = ({ children }: { children: React.ReactNode }) => {
  const { theme, toggleTheme } = useTheme();
  const { data: apiHealth, isError } = useQuery({
    queryKey: ["api-health"],
    queryFn: checkApiHealth,
    retry: 1,
    staleTime: 30_000,
  });
  const apiReachable = !!apiHealth && !isError;
  const apiHealthy = apiHealth?.status === "healthy";
  const statusClass = apiHealthy
    ? "text-emerald-400"
    : apiReachable
      ? "text-amber-400"
      : "text-muted-foreground";
  const dotClass = apiHealthy
    ? "bg-emerald-400 animate-pulse"
    : apiReachable
      ? "bg-amber-400"
      : "bg-muted-foreground";
  const statusText = apiHealthy
    ? "DATA LIVE"
    : apiReachable
      ? "API DEGRADED"
      : "API OFFLINE";

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Fixed header — full width, highest z-index */}
      <header className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-5 glass">
        {/* Left: CS branding */}
        <div className="flex items-center gap-2.5">
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center font-bold text-xs text-white shrink-0"
            style={{ background: 'var(--logo-gradient)' }}
          >
            CS
          </div>
          <span className="text-sm font-semibold text-foreground tracking-tight">CausalStress</span>
          <div className={`ml-3 hidden md:flex items-center gap-1.5 text-[10px] font-mono ${statusClass}`}>
            <div className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
            {statusText}
          </div>
        </div>

        {/* Right: actions */}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="w-9 h-9 rounded-xl border border-border bg-secondary/50 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <button className="relative w-9 h-9 rounded-xl border border-border bg-secondary/50 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors">
            <Bell className="w-4 h-4" />
            <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-red-500 ring-2 ring-background" />
          </button>
          <button className="w-9 h-9 rounded-full overflow-hidden shrink-0">
            <div className="w-full h-full flex items-center justify-center text-xs font-semibold text-white"
              style={{ background: 'var(--logo-gradient)' }}>
              <User className="w-4 h-4" />
            </div>
          </button>
        </div>
      </header>

      {/* Body: sidebar + content */}
      <div className="flex flex-1 pt-14">
        <Sidebar />
        <main className="flex-1 ml-[72px]">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
