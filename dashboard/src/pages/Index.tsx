import { useState, useEffect } from "react";
import {
  loadDashboardData,
  type DashboardData,
  type Elephant,
  type Prediction,
  type Village,
  type AlertEvent,
  type MovementPoint,
} from "@/lib/mock-data";
import ElephantSelector from "@/components/dashboard/ElephantSelector";
import ScenarioSelector, { type ScenarioMeta } from "@/components/dashboard/ScenarioSelector";
import CurrentStatePanel from "@/components/dashboard/CurrentStatePanel";
import TopPredictionPanel from "@/components/dashboard/TopPredictionPanel";
import RankedPredictionsPanel from "@/components/dashboard/RankedPredictionsPanel";
import VillagesPanel from "@/components/dashboard/VillagesPanel";
import AlertFeed from "@/components/dashboard/AlertFeed";
import MovementChart from "@/components/dashboard/MovementChart";
import StatsBar from "@/components/dashboard/StatsBar";
import ElephantMapPanel from "@/components/dashboard/ElephantMapPanel";
import StatusIndicator from "@/components/dashboard/StatusIndicator";

const ELEPHANT_COLORS = [
  "#4ECDC4", "#FF6B6B", "#FFE66D", "#A8E6CF", "#FF8E53",
  "#C3A6FF", "#56CFE1", "#FF9A9E",
];

const Index = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>("all");
  const [selectedRank, setSelectedRank] = useState<number | null>(1);
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [activeScenarioId, setActiveScenarioId] = useState<number | null>(null);

  // Load scenarios index on mount
  useEffect(() => {
    fetch("/scenarios/index.json")
      .then((res) => res.json())
      .then((s) => setScenarios(s))
      .catch((err) => console.error("Failed to load scenarios index", err));
  }, []);

  const loadScenario = (file: string, id: number) => {
    setLoading(true);
    loadDashboardData(id)
      .then((d) => {
        setData(d);
        setActiveScenarioId(id);
        setSelectedId("all");
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    // Initial load: live data from API
    loadDashboardData()
      .then((d) => {
        setData(d);
        setSelectedId("all");
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
        <div className="text-center space-y-3">
          <div className="text-3xl animate-pulse">🐘</div>
          <p className="text-sm text-muted-foreground">Loading predictions…</p>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
        <div className="text-center space-y-2">
          <p className="text-red-400 font-semibold">Failed to load prediction data</p>
          <p className="text-xs text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const elephants: Elephant[]        = data.elephants;
  const isAllView                    = selectedId === "all";
  
  // Data for the display
  const currentElephant = isAllView 
    ? elephants[0] // Fallback for components that need a single elephant but we are in all view
    : elephants.find((e) => e.id === selectedId) ?? elephants[0];

  const predictions = isAllView
    ? Object.values(data.predictionsMap).flat().sort((a, b) => b.confidence - a.confidence)
    : data.predictionsMap[selectedId!] ?? [];

  const villages = isAllView
    ? Object.values(data.villagesMap).flat().filter((v, i, self) => i === self.findIndex(t => t.name === v.name))
    : data.villagesMap[selectedId!] ?? [];

  const topPrediction = predictions[0];
  
  const movement = isAllView
    ? [] // We might want aggregate movement or just skip it for all view
    : data.movementMap?.[selectedId!] ?? [];

  const filteredAlerts = isAllView
    ? data.alertEvents
    : data.alertEvents.filter((a) => a.elephantId === selectedId);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              Elephant Tracking & Early Warning System
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              {activeScenarioId ? `Demo: ${scenarios.find(s => s.id === activeScenarioId)?.title}` : "Live Feed"} | {data.generatedAt}
            </p>
          </div>
          
          <div className="flex flex-col gap-2">
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">
              Select Demo Scenario
            </span>
            <ScenarioSelector
              scenarios={scenarios}
              activeId={activeScenarioId}
              onSelect={(sc) => loadScenario(sc.file, sc.id)}
            />
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-[11px] uppercase tracking-widest text-muted-foreground font-semibold">
              Tracked Elephants
            </span>
          </div>
          <ElephantSelector
            elephants={elephants}
            selectedId={selectedId || "all"}
            onSelect={(id) => {
              setSelectedId(id);
              setSelectedRank(1);
            }}
          />
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        {/* Stats */}
        <div className="mb-6">
          <StatsBar elephants={elephants} />
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Column (Requirement 2) */}
          <div className="space-y-6 lg:col-span-1">
            {!isAllView && <CurrentStatePanel elephant={currentElephant} />}
            {isAllView && (
               <div className="rounded-lg border border-border bg-card p-5">
                 <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">
                    Elephants Overview
                 </h3>
                 <div className="space-y-2">
                    {elephants.map(e => (
                      <div key={e.id} className="flex items-center justify-between text-sm">
                        <span>{e.name}</span>
                        <StatusIndicator status={e.status} size="sm" />
                      </div>
                    ))}
                 </div>
               </div>
            )}
            <MovementChart movement={movement} />
          </div>

          {/* Center Column (Requirement 3) */}
          <div className="space-y-6 lg:col-span-2">
            {topPrediction && (
              <TopPredictionPanel 
                prediction={topPrediction} 
                villages={villages} 
                allPredictions={predictions}
              />
            )}
            
            {/* Full-width Map for selected elephant (Requirement 5) */}
            <div className="mt-0">
              <ElephantMapPanel
                elephant={isAllView ? null as any : currentElephant}
                allElephants={isAllView ? elephants : undefined}
                predictions={predictions}
                historyPath={isAllView ? [] : (data.historyMap?.[selectedId!] ?? [])}
                color={isAllView ? "#4ECDC4" : ELEPHANT_COLORS[elephants.findIndex(e => e.id === selectedId) % ELEPHANT_COLORS.length]}
                allPredictions={isAllView ? data.predictionsMap : undefined}
                villages={villages}
              />
            </div>
          </div>

          {/* Right Column (Requirement 4) */}
          <div className="space-y-6 lg:col-span-1">
            <VillagesPanel villages={villages} />
            <AlertFeed
              alerts={filteredAlerts}
              onSelectElephant={(id) => {
                setSelectedId(id);
                setSelectedRank(1);
              }}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
