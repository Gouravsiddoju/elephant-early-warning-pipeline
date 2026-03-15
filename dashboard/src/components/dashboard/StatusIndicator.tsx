import { cn } from "@/lib/utils";

interface StatusIndicatorProps {
  status: "safe" | "warning" | "danger";
  size?: "sm" | "md" | "lg";
  pulse?: boolean;
}

const StatusIndicator = ({ status, size = "md", pulse = true }: StatusIndicatorProps) => {
  const sizeClasses = {
    sm: "h-2 w-2",
    md: "h-3 w-3",
    lg: "h-4 w-4",
  };

  const colorClasses = {
    safe: "bg-safe",
    warning: "bg-warning",
    danger: "bg-danger",
  };

  return (
    <span className="relative inline-flex">
      {pulse && status !== "safe" && (
        <span
          className={cn(
            "absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping",
            colorClasses[status]
          )}
        />
      )}
      <span
        className={cn(
          "relative inline-flex rounded-full",
          sizeClasses[size],
          colorClasses[status]
        )}
      />
    </span>
  );
};

export default StatusIndicator;
