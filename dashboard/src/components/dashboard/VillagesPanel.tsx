import type { Village } from "@/lib/mock-data";

interface VillagesPanelProps {
  villages: Village[];
}

const VillagesPanel = ({ villages }: VillagesPanelProps) => {
  const atRiskCount = villages.filter((v) => v.atRisk).length;

  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="h-2 w-2 rounded-sm bg-warning" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Villages ({atRiskCount} at risk)
        </h3>
      </div>

      {villages.length === 0 ? (
        <p className="text-sm italic text-muted-foreground">No villages within 5 km</p>
      ) : (
        <div className="space-y-2">
          {villages.map((village) => (
            <div
              key={village.name}
              className={`rounded-md px-3 py-2.5 border ${
                village.atRisk
                  ? "border-danger/30 bg-danger/5"
                  : "border-border bg-secondary/50"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">{village.name}</span>
                <span
                  className={`text-xs font-mono font-semibold ${
                    village.atRisk ? "text-danger" : "text-muted-foreground"
                  }`}
                >
                  {village.distanceKm} km
                </span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-[11px] text-muted-foreground">
                  Pop. {village.population.toLocaleString()}
                </span>
                {village.atRisk && (
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-danger">
                    At Risk
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default VillagesPanel;
