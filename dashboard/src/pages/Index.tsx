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

const ELEPHANT_COLORS = [
  "#4ECDC4", "#FF6B6B", "#FFE66D", "#A8E6CF", "#FF8E53",
  "#C3A6FF", "#56CFE1", "#FF9A9E",
];


const Index = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
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
        if (d.elephants.length > 0) setSelectedId(d.elephants[0].id);
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
        if (d.elephants.length > 0) setSelectedId(d.elephants[0].id);
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
  const currentId                     = selectedId ?? elephants[0]?.id;
  const elephantIdx                   = elephants.findIndex((e) => e.id === currentId);
  const elephantColor                 = ELEPHANT_COLORS[elephantIdx >= 0 ? elephantIdx : 0];
  const elephant: Elephant            = elephants.find((e) => e.id === currentId) ?? elephants[0];
  const predictions: Prediction[]     = data.predictionsMap[currentId] ?? [];
  const villages: Village[]           = data.villagesMap[currentId]    ?? [];
  const topPrediction: Prediction | undefined = predictions[0];
  const movement: MovementPoint[]     = data.movementMap?.[currentId]  ?? [];
  const historyPath: number[][]       = data.historyMap?.[currentId]   ?? [];
  const filteredAlerts: AlertEvent[]  = data.alertEvents.filter((a) => a.elephantId === currentId);

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
            selectedId={currentId}
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <CurrentStatePanel elephant={elephant} />
            {topPrediction && (
              <TopPredictionPanel prediction={topPrediction} villages={villages} />
            )}
            <VillagesPanel villages={villages} />
          </div>

          {/* Center Column */}
          <div className="space-y-6">
            <RankedPredictionsPanel
              predictions={predictions}
              selectedRank={selectedRank}
              onSelectRank={setSelectedRank}
            />
            <MovementChart movement={movement} />
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <AlertFeed
              alerts={filteredAlerts.length > 0 ? filteredAlerts : data.alertEvents}
              onSelectElephant={(id) => {
                setSelectedId(id);
                setSelectedRank(1);
              }}
            />
          </div>
        </div>

        {/* Full-width Map for selected elephant */}
        <div className="mt-6">
          <ElephantMapPanel
            elephant={elephant}
            predictions={predictions}
            historyPath={historyPath}
            color={elephantColor}
          />
        </div>
      </main>
    </div>
  );
};

export default Index;
