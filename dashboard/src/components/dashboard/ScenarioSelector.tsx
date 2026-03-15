interface ScenarioMeta {
  id: number;
  title: string;
  description: string;
  file: string;
}

// Status color for the scenario severity badge
const SCENARIO_BADGE: Record<number, string> = {
  1: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
  2: "bg-orange-500/20 text-orange-400 border-orange-500/40",
  3: "bg-yellow-500/20 text-yellow-400 border-yellow-500/40",
  4: "bg-red-500/20    text-red-400    border-red-500/40",
  5: "bg-amber-500/20  text-amber-400  border-amber-500/40",
};

interface ScenarioSelectorProps {
  scenarios: ScenarioMeta[];
  activeId: number | null;
  onSelect: (scenario: ScenarioMeta) => void;
}

const ScenarioSelector = ({ scenarios, activeId, onSelect }: ScenarioSelectorProps) => {
  return (
    <div className="flex flex-wrap gap-2">
      {scenarios.map((sc) => (
        <button
          key={sc.id}
          onClick={() => onSelect(sc)}
          title={sc.description}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[12px] font-medium transition-all ${
            activeId === sc.id
              ? `${SCENARIO_BADGE[sc.id]} shadow-sm scale-[1.03]`
              : "bg-card border-border text-muted-foreground hover:border-primary hover:text-foreground"
          }`}
        >
          {sc.title}
        </button>
      ))}
    </div>
  );
};

export default ScenarioSelector;
export type { ScenarioMeta };
