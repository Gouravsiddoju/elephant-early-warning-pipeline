import type { Elephant } from "@/lib/mock-data";

interface StatsBarProps {
  elephants: Elephant[];
}

const StatsBar = ({ elephants }: StatsBarProps) => {
  const safe = elephants.filter((e) => e.status === "safe").length;
  const warning = elephants.filter((e) => e.status === "warning").length;
  const danger = elephants.filter((e) => e.status === "danger").length;

  const stats = [
    { label: "Tracked", value: elephants.length, color: "text-foreground" },
    { label: "Safe", value: safe, color: "text-safe" },
    { label: "Warning", value: warning, color: "text-warning" },
    { label: "Critical", value: danger, color: "text-danger" },
  ];

  return (
    <div className="grid grid-cols-4 gap-3">
      {stats.map((s) => (
        <div key={s.label} className="rounded-lg border border-border bg-card px-4 py-3 text-center">
          <p className={`text-2xl font-mono font-bold ${s.color}`}>{s.value}</p>
          <p className="text-[11px] uppercase tracking-widest text-muted-foreground mt-0.5">
            {s.label}
          </p>
        </div>
      ))}
    </div>
  );
};

export default StatsBar;
