import type { AlertEvent } from "@/lib/mock-data";
import { motion } from "framer-motion";

interface AlertFeedProps {
  alerts: AlertEvent[];
  onSelectElephant: (id: string) => void;
}

const severityStyles = {
  low: "border-l-muted-foreground/40",
  medium: "border-l-warning",
  high: "border-l-danger",
};

const typeLabels: Record<AlertEvent["type"], string> = {
  movement: "MOV",
  boundary: "BND",
  proximity: "PRX",
  prediction: "PRD",
};

const AlertFeed = ({ alerts, onSelectElephant }: AlertFeedProps) => {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="h-2 w-2 rounded-sm bg-foreground/50" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Alert Feed
        </h3>
      </div>

      <div className="space-y-1.5 max-h-[320px] overflow-y-auto pr-1">
        {alerts.map((alert, i) => (
          <motion.button
            key={alert.id}
            onClick={() => onSelectElephant(alert.elephantId)}
            className={`w-full text-left rounded-r-md border-l-2 px-3 py-2 hover:bg-secondary/70 transition-colors ${severityStyles[alert.severity]}`}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-muted-foreground bg-secondary px-1.5 py-0.5 rounded">
                {typeLabels[alert.type]}
              </span>
              <span className="text-[11px] text-muted-foreground">{alert.timestamp}</span>
            </div>
            <p className="text-[12px] leading-relaxed text-secondary-foreground">
              {alert.message}
            </p>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

export default AlertFeed;
