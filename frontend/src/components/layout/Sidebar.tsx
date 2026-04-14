import { useLocation, Link } from "react-router-dom";
import {
  LayoutDashboard,
  GitBranch,
  Activity,
  FlaskConical,
  Shield,
  MessageSquare,
  Settings,
} from "lucide-react";

const navItems = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "Causal Graph", url: "/causal-graph", icon: GitBranch },
  { title: "Regimes", url: "/regimes", icon: Activity },
  { title: "Scenario Lab", url: "/scenarios", icon: FlaskConical },
  { title: "Stress Test", url: "/stress-test", icon: Shield },
  { title: "AI Advisor", url: "/ai-advisor", icon: MessageSquare },
];

const Sidebar = () => {
  const location = useLocation();

  const isActive = (url: string) =>
    url === "/" ? location.pathname === "/" : location.pathname.startsWith(url);

  return (
    <div
      className="fixed left-0 top-14 z-40 flex flex-col w-[72px]"
      style={{
        height: 'calc(100vh - 3.5rem)',
        backgroundColor: 'var(--sidebar-bg-raw)',
        borderRight: '1px solid var(--sidebar-border-raw)',
      }}
    >
      {/* Nav */}
      <nav className="flex-1 flex flex-col gap-1 px-3 mt-3">
        {navItems.map((item) => {
          const active = isActive(item.url);
          const Icon = item.icon;
          return (
            <Link
              key={item.url}
              to={item.url}
              title={item.title}
              className={`flex items-center justify-center w-full h-11 rounded-xl transition-all duration-150 relative ${
                active
                  ? "bg-white/10 text-white"
                  : "text-white/35 hover:text-white/65 hover:bg-white/5"
              }`}
            >
              {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r" style={{ backgroundColor: 'var(--sidebar-active-accent)' }} />
              )}
              <Icon className="w-5 h-5" />
            </Link>
          );
        })}
      </nav>

      {/* Bottom */}
      <div className="px-3 pb-4 flex flex-col items-center gap-1">
        <Link
          to="/settings"
          title="Settings"
          className={`flex items-center justify-center w-full h-11 rounded-xl transition-all duration-150 relative ${
            isActive("/settings")
              ? "bg-white/10 text-white"
              : "text-white/35 hover:text-white/65 hover:bg-white/5"
          }`}
        >
          {isActive("/settings") && (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r" style={{ backgroundColor: 'var(--sidebar-active-accent)' }} />
          )}
          <Settings className="w-5 h-5" />
        </Link>
        <span className="text-[9px] text-white/15 font-mono mt-1">v0.1</span>
      </div>
    </div>
  );
};

export default Sidebar;
