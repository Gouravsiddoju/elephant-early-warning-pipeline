import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { MovementPoint } from "@/lib/mock-data";

interface MovementChartProps {
  movement: MovementPoint[];
}

const MovementChart = ({ movement }: MovementChartProps) => {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="h-2 w-2 rounded-sm bg-primary" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Movement Activity (Last 24 steps)
        </h3>
      </div>

      <div className="h-[180px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={movement} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="moveGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(152, 45%, 45%)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="hsl(152, 45%, 45%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(200, 12%, 18%)" />
            <XAxis
              dataKey="hour"
              tick={{ fontSize: 9, fill: "hsl(200, 10%, 50%)" }}
              axisLine={{ stroke: "hsl(200, 12%, 18%)" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(200, 10%, 50%)" }}
              axisLine={false}
              tickLine={false}
              unit=" km"
            />
            <Tooltip
              contentStyle={{
                background: "hsl(200, 15%, 11%)",
                border: "1px solid hsl(200, 12%, 18%)",
                borderRadius: "6px",
                fontSize: "12px",
                color: "hsl(180, 10%, 85%)",
              }}
              formatter={(v: number) => [`${v.toFixed(3)} km`, "Distance"]}
            />
            <Area
              type="monotone"
              dataKey="distance"
              stroke="hsl(152, 45%, 45%)"
              strokeWidth={2}
              fill="url(#moveGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MovementChart;
