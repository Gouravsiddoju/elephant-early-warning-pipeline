import type { Prediction } from "@/lib/mock-data";
import { motion } from "framer-motion";

interface RankedPredictionsPanelProps {
  predictions: Prediction[];
  selectedRank: number | null;
  onSelectRank: (rank: number) => void;
}

const RankedPredictionsPanel = ({
  predictions,
  selectedRank,
  onSelectRank,
}: RankedPredictionsPanelProps) => {
  const maxConfidence = Math.max(...predictions.map((p) => p.confidence));

  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="h-2 w-2 rounded-sm bg-info" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Ranked Predictions
        </h3>
      </div>

      <div className="space-y-2">
        {predictions.map((pred) => (
          <motion.button
            key={pred.rank}
            onClick={() => onSelectRank(pred.rank)}
            className={`w-full text-left rounded-md px-3 py-2.5 transition-colors border ${
              selectedRank === pred.rank
                ? "border-primary/40 bg-primary/10"
                : "border-transparent hover:bg-secondary"
            }`}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2.5">
                <span
                  className={`flex h-5 w-5 items-center justify-center rounded text-[11px] font-bold ${
                    pred.rank === 1
                      ? "bg-danger/20 text-danger"
                      : pred.rank === 2
                      ? "bg-warning/20 text-warning"
                      : "bg-secondary text-muted-foreground"
                  }`}
                >
                  {pred.rank}
                </span>
                <span className="text-sm font-mono font-medium text-foreground">
                  {pred.gridCell}
                </span>
              </div>
              <span className="text-sm font-mono font-semibold text-primary">
                {pred.confidence.toFixed(2)}%
              </span>
            </div>

            <div className="ml-[30px]">
              <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-primary/60"
                  initial={{ width: 0 }}
                  animate={{ width: `${(pred.confidence / maxConfidence) * 100}%` }}
                  transition={{ duration: 0.6, delay: pred.rank * 0.1 }}
                />
              </div>
              <p className="text-[11px] text-muted-foreground mt-1">
                {pred.location.lat.toFixed(4)}, {pred.location.lng.toFixed(4)} | {pred.distanceKm} km away
              </p>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

export default RankedPredictionsPanel;
