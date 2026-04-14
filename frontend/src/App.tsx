import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Layout from "@/components/layout/Layout";
import Index from "./pages/Index.tsx";
import CausalGraph from "./pages/CausalGraph.tsx";
import Regimes from "./pages/Regimes.tsx";
import ScenarioLab from "./pages/ScenarioLab.tsx";
import StressTest from "./pages/StressTest.tsx";
import AIAdvisor from "./pages/AIAdvisor.tsx";
import NotFound from "./pages/NotFound.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/causal-graph" element={<CausalGraph />} />
              <Route path="/regimes" element={<Regimes />} />
              <Route path="/scenarios" element={<ScenarioLab />} />
              <Route path="/stress-test" element={<StressTest />} />
              <Route path="/ai-advisor" element={<AIAdvisor />} />
              <Route path="/settings" element={<div className="p-8 text-center text-muted-foreground">Settings — Coming Soon</div>} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
