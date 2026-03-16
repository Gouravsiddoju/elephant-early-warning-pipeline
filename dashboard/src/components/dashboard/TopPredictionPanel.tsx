import { useState } from "react";
import type { Prediction, Village } from "@/lib/mock-data";
import { ChevronDown, ChevronUp, BarChart2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface TopPredictionPanelProps {
  prediction: Prediction;
  villages: Village[];
  allPredictions?: Prediction[]; // Added to support expansion
}

const TopPredictionPanel = ({ prediction, villages, allPredictions = [] }: TopPredictionPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const atRiskCount = villages.filter((v) => v.atRisk).length;
  const isSafe = atRiskCount === 0;

  const maxConfidence = allPredictions.length > 0 
    ? Math.max(...allPredictions.map((p) => p.confidence))
    : prediction.confidence;

  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-danger" />
            <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
              Top Prediction
            </h3>
          </div>
          {allPredictions.length > 0 && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-1.5 px-2 py-1 text-[10px] font-medium rounded bg-secondary/50 text-muted-foreground hover:text-foreground transition-colors border border-border/50"
            >
              <BarChart2 className="w-3 h-3" />
              {isExpanded ? "Hide Ranked" : "Show Ranked"}
              {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          )}
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

      <AnimatePresence>
        {isExpanded && allPredictions.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="border-t border-border bg-muted/30"
          >
            <div className="p-5 space-y-2 max-h-[400px] overflow-y-auto">
              <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold mb-3">
                All Ranked Predictions
              </h4>
              {allPredictions.map((pred) => (
                <div
                  key={pred.rank}
                  className={`rounded-md px-3 py-2.5 transition-colors border bg-card/50 ${
                    pred.rank === 1 ? "border-primary/20" : "border-border/40"
                  }`}
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
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default TopPredictionPanel;
