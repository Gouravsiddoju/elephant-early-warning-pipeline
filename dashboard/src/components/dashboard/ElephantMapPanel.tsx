import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import type { Elephant, Prediction } from "@/lib/mock-data";

import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl;
L.Icon.Default.mergeOptions({ iconUrl, iconRetinaUrl, shadowUrl });

const RANK_COLORS = ["#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#007AFF"];

const TILE_LAYERS: Record<string, { url: string; label: string; attribution: string }> = {
  dark:      { label: "Dark",      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",              attribution: "&copy; CARTO" },
  satellite: { label: "Satellite", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attribution: "&copy; Esri" },
  street:    { label: "Street",    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",                         attribution: "&copy; OpenStreetMap contributors" },
  light:     { label: "Light",     url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",             attribution: "&copy; CARTO" },
};

interface ElephantMapPanelProps {
  elephant: Elephant;
  predictions: Prediction[];
  historyPath?: number[][];
  color?: string;
}

const ElephantMapPanel = ({
  elephant,
  predictions,
  historyPath = [],
  color = "#4ECDC4",
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

  // Layer group refs so we can show/hide without rebuilding the map
  const pathLayerRef    = useRef<L.Polyline | null>(null);
  const arrowLayerRef   = useRef<L.LayerGroup | null>(null);   // dashed line + top circle
  const zonesLayerRef   = useRef<L.LayerGroup | null>(null);   // all prediction polygons
  const historyLayerRef = useRef<L.Polyline | null>(null);

  // ── Build map once per elephant
  useEffect(() => {
    if (!containerRef.current) return;

    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    const { lat, lng } = elephant.position;
    const map = L.map(containerRef.current, { zoomControl: true, attributionControl: true });
    mapRef.current = map;

    // Tile layer
    const tileConfig = TILE_LAYERS[activeLayer];
    tileRef.current = L.tileLayer(tileConfig.url, { maxZoom: 18, attribution: tileConfig.attribution }).addTo(map);

    const bounds: L.LatLng[] = [];

    // ── Current position marker (always visible)
    const curIcon = L.divIcon({
      className: "",
      html: `<div style="background:${color};border:3px solid white;border-radius:50%;
             width:32px;height:32px;display:flex;align-items:center;justify-content:center;
             font-size:16px;box-shadow:0 0 14px ${color}bb;cursor:pointer">🐘</div>`,
      iconSize: [32, 32], iconAnchor: [16, 16],
    });
    L.marker([lat, lng], { icon: curIcon })
      .bindPopup(`<b>${elephant.name}</b><br>Grid: <code>${elephant.gridCell}</code><br>${lat.toFixed(4)}°, ${lng.toFixed(4)}°`)
      .addTo(map);
    bounds.push(L.latLng(lat, lng));

    // ── Historical dead-reckoned path
    if (historyPath.length >= 2) {
      const poly = L.polyline(historyPath as L.LatLngExpression[], {
        color, weight: 3, opacity: 0.9, dashArray: "1 0",
      }).addTo(map);
      historyLayerRef.current = poly;
      historyPath.forEach((pt) => bounds.push(L.latLng(pt[0], pt[1])));
    }

    // ── Prediction zones (polygons for all ranked grids)
    const zonesGroup = L.layerGroup().addTo(map);
    zonesLayerRef.current = zonesGroup;
    predictions.forEach((pred, i) => {
      const plat = pred.location.lat;
      const plon = pred.location.lng;
      if (!plat || !plon) return;
      const rc = RANK_COLORS[i] ?? "#FFFFFF";
      const half = 0.023;
      L.polygon(
        [[plat - half, plon - half],[plat - half, plon + half],
         [plat + half, plon + half],[plat + half, plon - half]] as L.LatLngExpression[],
        { color: rc, weight: 2, fill: true, fillColor: rc, fillOpacity: 0.22 - i * 0.03 }
      ).bindPopup(`<b>Rank #${pred.rank}</b><br>Grid: <code>${pred.gridCell}</code><br>Confidence: ${pred.confidence.toFixed(2)}%`)
       .addTo(zonesGroup);
      bounds.push(L.latLng(plat, plon));
    });

    // ── Prediction arrow (line + circle for rank-1)
    const arrowGroup = L.layerGroup().addTo(map);
    arrowLayerRef.current = arrowGroup;
    if (predictions.length > 0) {
      const top = predictions[0];
      const plat = top.location.lat, plon = top.location.lng;
      if (plat && plon) {
        L.polyline([[lat, lng],[plat, plon]] as L.LatLngExpression[], {
          color: RANK_COLORS[0], weight: 3, opacity: 1, dashArray: "8 5",
        }).addTo(arrowGroup);
        L.circleMarker([plat, plon] as L.LatLngExpression, {
          radius: 8, color: "white", fillColor: RANK_COLORS[0], fillOpacity: 1, weight: 2,
        }).bindTooltip(`#1: ${top.gridCell} (${top.confidence.toFixed(1)}%)`)
         .addTo(arrowGroup);

        // Start dot at elephant
        L.circleMarker([lat, lng] as L.LatLngExpression, {
          radius: 5, color: "white", fillColor: color, fillOpacity: 1, weight: 2,
        }).addTo(arrowGroup);
      }
    }

    // ── Path (solid history line label)
    if (historyPath.length >= 2) {
      const pathLine = L.polyline(historyPath as L.LatLngExpression[], {
        color, weight: 4, opacity: 0.7,
      }).bindTooltip("Historical path");
      pathLayerRef.current = pathLine;
    }

    if (bounds.length > 0) map.fitBounds(L.latLngBounds(bounds), { padding: [32, 32] });
    else map.setView([lat, lng], 10);

    return () => { map.remove(); mapRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elephant.id]);

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
      {/* ── Top bar: Basemap toggles */}
      <div className="flex items-center gap-2 px-5 py-2.5 border-b border-border flex-wrap">
        <span className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mr-auto">
          Live Map — Elephant {elephant.id}
        </h3>
        <span className="text-[10px] text-muted-foreground font-mono">{elephant.gridCell}</span>
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
