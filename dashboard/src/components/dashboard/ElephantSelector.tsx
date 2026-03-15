import type { Elephant } from "@/lib/mock-data";
import StatusIndicator from "./StatusIndicator";

interface ElephantSelectorProps {
  elephants: Elephant[];
  selectedId: string;
  onSelect: (id: string) => void;
}

const ElephantSelector = ({ elephants, selectedId, onSelect }: ElephantSelectorProps) => {
  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {elephants.map((el) => (
        <button
          key={el.id}
          onClick={() => onSelect(el.id)}
          className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-all border ${
            selectedId === el.id
              ? "border-primary/40 bg-primary/10 text-foreground"
              : "border-border bg-card text-muted-foreground hover:bg-secondary hover:text-foreground"
          }`}
        >
          <StatusIndicator status={el.status} size="sm" pulse={selectedId === el.id} />
          <span className="font-mono text-xs">{el.id}</span>
          <span className="text-xs">{el.name}</span>
        </button>
      ))}
    </div>
  );
};

export default ElephantSelector;
