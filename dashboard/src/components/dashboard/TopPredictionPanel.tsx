import type { Prediction, Village } from "@/lib/mock-data";

interface TopPredictionPanelProps {
  prediction: Prediction;
  villages: Village[];
}

const TopPredictionPanel = ({ prediction, villages }: TopPredictionPanelProps) => {
  const atRiskCount = villages.filter((v) => v.atRisk).length;
  const isSafe = atRiskCount === 0;

  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="h-2 w-2 rounded-full bg-danger" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Top Prediction
        </h3>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Grid Cell</span>
          <span className="text-sm font-mono font-medium text-foreground">
            {prediction.gridCell}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Confidence</span>
          <span className="text-sm font-mono font-semibold text-primary">
            {prediction.confidence.toFixed(2)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Location</span>
          <span className="text-sm font-mono font-medium text-foreground">
            {prediction.location.lat.toFixed(4)}, {prediction.location.lng.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Distance</span>
          <span className="text-sm font-mono font-medium text-foreground">
            {prediction.distanceKm} km
          </span>
        </div>

        <div
          className={`mt-2 rounded-md px-3 py-2 text-sm font-medium ${
            isSafe
              ? "bg-safe/10 text-safe border border-safe/20"
              : "bg-danger/10 text-danger border border-danger/20"
          }`}
        >
          {isSafe
            ? "SAFE -- No settlements in path"
            : `WARNING -- ${atRiskCount} village${atRiskCount > 1 ? "s" : ""} at risk`}
        </div>
      </div>
    </div>
  );
};

export default TopPredictionPanel;
