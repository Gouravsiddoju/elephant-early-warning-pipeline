import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import type { Elephant, Prediction, Village } from "@/lib/mock-data";

import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl;
L.Icon.Default.mergeOptions({ iconUrl, iconRetinaUrl, shadowUrl });

const ELEPHANT_COLORS = [
  "#4ECDC4", "#FF6B6B", "#FFE66D", "#A8E6CF", "#FF8E53",
  "#C3A6FF", "#56CFE1", "#FF9A9E",
];

const RANK_COLORS = ["#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#007AFF"];

const TILE_LAYERS: Record<string, { url: string; label: string; attribution: string }> = {
  dark:      { label: "Dark",      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",              attribution: "&copy; CARTO" },
  satellite: { label: "Satellite", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attribution: "&copy; Esri" },
  street:    { label: "Street",    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",                         attribution: "&copy; OpenStreetMap contributors" },
  light:     { label: "Light",     url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",             attribution: "&copy; CARTO" },
};

interface ElephantMapPanelProps {
  elephant?: Elephant | null;
  allElephants?: Elephant[];
  predictions: Prediction[];
  historyPath?: number[][];
  color?: string;
  allPredictions?: Record<string, Prediction[]>;
  villages?: Village[];
}

const ElephantMapPanel = ({
  elephant,
  allElephants = [],
  predictions,
  historyPath = [],
  color = "#4ECDC4",
  allPredictions = {},
  villages = [],
}: ElephantMapPanelProps) => {
  const mapRef        = useRef<L.Map | null>(null);
  const tileRef       = useRef<L.TileLayer | null>(null);
  const containerRef  = useRef<HTMLDivElement>(null);

  // ── Basemap state
  const [activeLayer, setActiveLayer] = useState<string>("dark");

  // ── Overlay toggle state
  const [showPath,    setShowPath]    = useState(true);
  const [showZones,   setShowZones]   = useState(true);
  const [showArrow,   setShowArrow]   = useState(true);
  const [showHistory, setShowHistory] = useState(true);
  const [showVillages, setShowVillages] = useState(true);

  // Layer group refs so we can show/hide without rebuilding the map
  const pathLayerRef    = useRef<L.Polyline | null>(null);
  const arrowLayerRef   = useRef<L.LayerGroup | null>(null);   // dashed line + top circle
  const zonesLayerRef   = useRef<L.LayerGroup | null>(null);   // all prediction polygons
  const historyLayerRef = useRef<L.Polyline | null>(null);
  const villagesLayerRef = useRef<L.LayerGroup | null>(null);

  // ── Build map once per elephant / view
  useEffect(() => {
    if (!containerRef.current) return;

    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    const map = L.map(containerRef.current, { zoomControl: true, attributionControl: true });
    mapRef.current = map;

    // Tile layer
    const tileConfig = TILE_LAYERS[activeLayer];
    tileRef.current = L.tileLayer(tileConfig.url, { maxZoom: 18, attribution: tileConfig.attribution }).addTo(map);

    const bounds: L.LatLng[] = [];

    // ── Markers logic
    const renderElephant = (el: Elephant, isMain: boolean, displayColor: string) => {
      const { lat, lng } = el.position;
      const curIcon = L.divIcon({
        className: "",
        html: `<div style="background:${displayColor};border:3px solid white;border-radius:50%;
               width:${isMain ? 32 : 24}px;height:${isMain ? 32 : 24}px;display:flex;align-items:center;justify-content:center;
               font-size:${isMain ? 16 : 12}px;box-shadow:0 0 14px ${displayColor}bb;cursor:pointer">🐘</div>`,
        iconSize: [isMain ? 32 : 24, isMain ? 32 : 24], 
        iconAnchor: [isMain ? 16 : 12, isMain ? 16 : 12],
      });
      L.marker([lat, lng], { icon: curIcon })
        .bindPopup(`<b>${el.name}</b><br>Grid: <code>${el.gridCell}</code><br>${lat.toFixed(4)}°, ${lng.toFixed(4)}°`)
        .addTo(map);
      bounds.push(L.latLng(lat, lng));
    };

    // ── Helper: Prediction Rendering
    const renderPredictionLines = (
      elPos: { lat: number; lng: number },
      elPredictions: Prediction[],
      elColor: string,
      group: L.LayerGroup
    ) => {
      elPredictions.slice(0, 5).forEach((pred, i) => {
        const plat = pred.location.lat;
        const plon = pred.location.lng;
        if (!plat || !plon) return;

        const isTop = i === 0;
        const rankColor = RANK_COLORS[i % RANK_COLORS.length] ?? "#FFFFFF";

        // Line from elephant to prediction
        L.polyline([[elPos.lat, elPos.lng], [plat, plon]] as L.LatLngExpression[], {
          color: isTop ? rankColor : elColor,
          weight: isTop ? 3 : 2,
          opacity: isTop ? 1 : 0.6,
          dashArray: isTop ? "" : "4 8",
        }).addTo(group);

        // Individual prediction circle
        L.circleMarker([plat, plon] as L.LatLngExpression, {
          radius: isTop ? 7 : 4,
          color: "white",
          fillColor: rankColor,
          fillOpacity: 1,
          weight: 1.5,
        }).bindTooltip(`#${pred.rank}: ${pred.gridCell} (${pred.confidence.toFixed(1)}%)`)
         .addTo(group);

        // Add to bounds
        bounds.push(L.latLng(plat, plon));

        // Ensure zone (polygon) is added
        const half = 0.023;
        L.polygon(
          [[plat - half, plon - half],[plat - half, plon + half],
           [plat + half, plon + half],[plat + half, plon - half]] as L.LatLngExpression[],
          { 
            color: rankColor, 
            weight: isTop ? 2 : 1, 
            fill: true, 
            fillColor: rankColor, 
            fillOpacity: isTop ? 0.22 : 0.1 
          }
        ).bindPopup(`<b>Rank #${pred.rank}</b><br>Grid: <code>${pred.gridCell}</code><br>Confidence: ${pred.confidence.toFixed(2)}%`)
         .addTo(zonesGroup);
      });
    };

    if (elephant) {
      renderElephant(elephant, true, color);
      
      // ── Historical dead-reckoned path (only for single select)
      if (historyPath.length >= 2) {
        const poly = L.polyline(historyPath as L.LatLngExpression[], {
          color, weight: 3, opacity: 0.9, dashArray: "1 0",
        }).addTo(map);
        historyLayerRef.current = poly;
        historyPath.forEach((pt) => bounds.push(L.latLng(pt[0], pt[1])));
      }
    } else if (allElephants.length > 0) {
      allElephants.forEach((el, idx) => {
        const elColor = ELEPHANT_COLORS[idx % ELEPHANT_COLORS.length];
        renderElephant(el, false, elColor);
      });
    }

    // ── Villages logic
    const villagesGroup = L.layerGroup().addTo(map);
    villagesLayerRef.current = villagesGroup;

    villages.forEach((v) => {
      if (!v.position) return;
      const { lat: vLat, lng: vLng } = v.position;
      const isAtRisk = v.distanceKm < 5;
      const villageIcon = L.divIcon({
        className: "village-marker-icon" + (isAtRisk ? " village-at-risk-icon" : ""),
        html: `<div style="background:#FFF;border:3px solid ${isAtRisk ? '#FF3B30' : '#555'};border-radius:50%;
               width:40px;height:40px;display:flex;align-items:center;justify-content:center;
               font-size:20px;box-shadow:0 0 15px rgba(0,0,0,0.4);cursor:pointer;position:relative">
                 <span style="position:relative;z-index:2">🏘️</span>
                 ${isAtRisk ? '<div style="position:absolute;inset:-4px;border-radius:50%;border:2px solid #FF3B30;opacity:0.6"></div>' : ''}
               </div>`,
        iconSize: [40, 40],
        iconAnchor: [20, 20],
      });
      L.marker([vLat, vLng], { icon: villageIcon })
        .bindPopup(`<b>${v.name}</b><br>Type: ${v.type || 'Village'}<br>${isAtRisk ? '<b>WARNING: RAID RISK</b><br>' : ''}Distance to Prediction: ${v.distanceKm} km`)
        .addTo(villagesGroup);

      if (isAtRisk) {
        L.circle([vLat, vLng], {
          radius: 1500, color: "#FF3B30", weight: 4, fill: true, fillColor: "#FF3B30", fillOpacity: 0.2, className: "glowing-zone"
        }).addTo(villagesGroup);
      }
      bounds.push(L.latLng(vLat, vLng));
    });

    // ── Prediction zones & arrows
    const zonesGroup = L.layerGroup().addTo(map);
    zonesLayerRef.current = zonesGroup;
    const arrowGroup = L.layerGroup().addTo(map);
    arrowLayerRef.current = arrowGroup;

    if (elephant) {
      // Single view: Show top 5 paths for THIS elephant
      renderPredictionLines(elephant.position, predictions, color, arrowGroup);
    } else {
      // Overview mode: Show top prediction path for EVERY elephant
      allElephants.forEach((el, idx) => {
        const elColor = ELEPHANT_COLORS[idx % ELEPHANT_COLORS.length];
        const elPreds = allPredictions[el.id] || [];
        if (elPreds.length > 0) {
          // In overview, we only show top rank to keep it clean, but for all cows
          renderPredictionLines(el.position, elPreds.slice(0, 1), elColor, arrowGroup);
        }
      });
    }

    if (bounds.length > 0) map.fitBounds(L.latLngBounds(bounds), { padding: [32, 32] });
    else if (elephant) map.setView([elephant.position.lat, elephant.position.lng], 10);

    return () => { map.remove(); mapRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elephant?.id, allElephants.length, predictions.length, villages.length]);

  // ── Swap basemap
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (tileRef.current) map.removeLayer(tileRef.current);
    const cfg = TILE_LAYERS[activeLayer];
    tileRef.current = L.tileLayer(cfg.url, { maxZoom: 18, attribution: cfg.attribution }).addTo(map);
    tileRef.current.bringToBack();
  }, [activeLayer]);

  // ── Toggle overlays
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !historyLayerRef.current) return;
    showHistory ? historyLayerRef.current.addTo(map) : map.removeLayer(historyLayerRef.current);
  }, [showHistory]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !zonesLayerRef.current) return;
    showZones ? zonesLayerRef.current.addTo(map) : map.removeLayer(zonesLayerRef.current);
  }, [showZones]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !arrowLayerRef.current) return;
    showArrow ? arrowLayerRef.current.addTo(map) : map.removeLayer(arrowLayerRef.current);
  }, [showArrow]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !pathLayerRef.current) return;
    showPath ? pathLayerRef.current.addTo(map) : map.removeLayer(pathLayerRef.current);
  }, [showPath]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !villagesLayerRef.current) return;
    showVillages ? villagesLayerRef.current.addTo(map) : map.removeLayer(villagesLayerRef.current);
  }, [showVillages]);

  // ── Helper: overlay toggle button
  const OverlayToggle = ({
    label, active, color: btnColor, onClick,
  }: { label: string; active: boolean; color: string; onClick: () => void }) => (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded border transition-all ${
        active
          ? "text-foreground border-border bg-muted/60"
          : "text-muted-foreground border-transparent bg-transparent opacity-50"
      }`}
    >
      <span
        className="h-2 w-2 rounded-full flex-shrink-0"
        style={{ background: active ? btnColor : "#555" }}
      />
      {label}
    </button>
  );

  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <style>{`
        @keyframes village-glow {
          0% { box-shadow: 0 0 10px rgba(255, 59, 48, 0.4); border-color: rgba(255, 59, 48, 0.6); }
          50% { box-shadow: 0 0 25px rgba(255, 59, 48, 0.9); border-color: rgba(255, 59, 48, 1); }
          100% { box-shadow: 0 0 10px rgba(255, 59, 48, 0.4); border-color: rgba(255, 59, 48, 0.6); }
        }
        @keyframes zone-pulse {
          0% { fill-opacity: 0.15; stroke-opacity: 0.4; }
          50% { fill-opacity: 0.35; stroke-opacity: 0.8; }
          100% { fill-opacity: 0.15; stroke-opacity: 0.4; }
        }
        .village-at-risk-icon {
          animation: village-glow 2s infinite ease-in-out;
        }
        .glowing-zone {
          animation: zone-pulse 3s infinite ease-in-out;
        }
      `}</style>
      {/* ── Top bar: Basemap toggles */}
      <div className="flex items-center gap-2 px-5 py-2.5 border-b border-border flex-wrap">
        <span className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mr-auto">
          {elephant ? `Live Map — Elephant ${elephant.id}` : "Live Map — All Elephants"}
        </h3>
        {elephant && (
          <span className="text-[10px] text-muted-foreground font-mono">{elephant.gridCell}</span>
        )}
        {/* Basemap selector */}
        <div className="flex gap-1 ml-2 border-l border-border pl-2">
          {Object.entries(TILE_LAYERS).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => setActiveLayer(key)}
              className={`px-2 py-0.5 text-[10px] font-medium rounded border transition-colors ${
                activeLayer === key
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-transparent text-muted-foreground border-border hover:border-primary hover:text-foreground"
              }`}
            >
              {cfg.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Layer overlay toggles */}
      <div className="flex items-center gap-1 px-5 py-2 border-b border-border bg-muted/20 flex-wrap">
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground mr-1">Layers:</span>
        <OverlayToggle label="Historical Path" active={showHistory} color={color}      onClick={() => setShowHistory((v) => !v)} />
        <OverlayToggle label="Prediction Zones" active={showZones} color="#FF3B30"     onClick={() => setShowZones((v) => !v)} />
        <OverlayToggle label="Direction Arrow"  active={showArrow} color={RANK_COLORS[0]} onClick={() => setShowArrow((v) => !v)} />
        <OverlayToggle label="Track Line"       active={showPath}  color={color}       onClick={() => setShowPath((v) => !v)} />
        <OverlayToggle label="Villages"         active={showVillages} color="#FFF"      onClick={() => setShowVillages((v) => !v)} />
      </div>

      {/* ── Legend */}
      <div className="flex items-center gap-4 px-5 py-1.5 border-b border-border bg-card/50 flex-wrap">
        {predictions.slice(0, 5).map((p, i) => (
          <div key={p.gridCell} className="flex flex-col gap-1">
            <div className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm flex-shrink-0" style={{ background: RANK_COLORS[i] }} />
              <span className="text-[10px] text-muted-foreground font-mono">
                #{p.rank} {p.gridCell} <span className="text-foreground">{p.confidence.toFixed(1)}%</span>
              </span>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-center">
                <div className="text-xs font-mono text-muted-foreground">
                  {p.location.lat.toFixed(4)}, {p.location.lng.toFixed(4)} | {p.distanceKm} km away
                </div>
              </div>
              {p.reasoning && p.reasoning.length > 0 && (
                <div className="mt-2 text-[10px] space-y-1">
                  {p.reasoning.map((reason, idx) => (
                    <div key={idx} className="flex gap-2 text-muted-foreground/80 leading-tight">
                      <div className="w-1 h-1 rounded-full bg-blue-400 mt-1 shrink-0" />
                      <span>{reason}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ── Map */}
      <div ref={containerRef} style={{ height: 460, width: "100%" }} />
    </div>
  );
};

export default ElephantMapPanel;
