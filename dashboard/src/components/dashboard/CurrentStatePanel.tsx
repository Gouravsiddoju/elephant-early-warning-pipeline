import type { Elephant } from "@/lib/mock-data";
import StatusIndicator from "./StatusIndicator";

interface CurrentStatePanelProps {
  elephant: Elephant;
}

const CurrentStatePanel = ({ elephant }: CurrentStatePanelProps) => {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <StatusIndicator status={elephant.status} size="sm" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Current State
        </h3>
      </div>

      <div className="space-y-3">
        <DataRow label="Elephant ID" value={elephant.id} />
        <DataRow label="Name" value={elephant.name} />
        <DataRow label="Grid Cell" value={elephant.gridCell} mono />
        <DataRow
          label="Position"
          value={`${elephant.position.lat.toFixed(4)}, ${elephant.position.lng.toFixed(4)}`}
          mono
        />
        <DataRow label="Horizon" value={`${elephant.horizon} hours`} />
        <DataRow label="Model" value={elephant.model} />
        <div className="pt-2 border-t border-border">
          <p className="text-[11px] text-muted-foreground">
            Last updated: {elephant.lastUpdated}
          </p>
        </div>
      </div>
    </div>
  );
};

function DataRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={`text-sm font-medium text-foreground ${mono ? "font-mono" : ""}`}>
        {value}
      </span>
    </div>
  );
}

export default CurrentStatePanel;
