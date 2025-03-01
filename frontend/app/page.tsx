"use client";

import React, { useState } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type TokenData = {
  text: string;
  prob: number;
  log_prob: number;
  entropy: number;
  top_tokens: string[];
  top_logits: number[];
};

type ColorScheme = "blueOrange" | "greenRed";

function mapValueToColor(
  value: number, 
  min: number, 
  max: number, 
  scheme: ColorScheme,
  colorMode: "prob" | "entropy"
): string {
  const norm = max > min ? (value - min) / (max - min) : 0.5;
  
  switch (scheme) {
    case "blueOrange":
      const adjustedNorm = colorMode === "prob" ? (1 - norm) : norm;
      const r = Math.round(255 * adjustedNorm);
      const b = Math.round(255 * (1 - adjustedNorm));
      return `rgb(${r}, ${Math.round(r * 0.65)}, ${b})`;
    
    case "greenRed":
      const g = Math.round(255 * (1 - norm));
      const r2 = Math.round(255 * norm);
      return `rgb(${r2}, ${g}, 0)`;
      
    default:
      return `rgb(0, 0, 0)`;
  }
}

function generateColorScale(scheme: ColorScheme, colorMode: "prob" | "entropy"): string[] {
  const steps = 10;
  const colors = [];
  for (let i = 0; i < steps; i++) {
    const norm = i / (steps - 1);
    colors.push(mapValueToColor(norm, 0, 1, scheme, colorMode));
  }
  return colors;
}

function ColorLegend({ 
  scheme, 
  min, 
  max, 
  label,
  colorMode 
}: { 
  scheme: ColorScheme; 
  min: number; 
  max: number; 
  label: string;
  colorMode: "prob" | "entropy";
}) {
  const colors = generateColorScale(scheme, colorMode);
  
  return (
    <div className="flex items-center gap-4 p-2 border rounded-md">
      <div className="flex items-center gap-2">
        <div className="h-[20px] w-[200px] relative">
          <div className="absolute inset-0 flex">
            {colors.map((color, i) => (
              <div
                key={i}
                style={{ backgroundColor: color }}
                className="flex-1"
              />
            ))}
          </div>
        </div>
      </div>
      <div className="flex justify-between items-center gap-4 text-sm min-w-[150px]">
        <span>{min.toFixed(2)}</span>
        <span className="font-medium">{label}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  );
}

export default function Page() {
  const [prompt, setPrompt] = useState("");
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [fullText, setFullText] = useState("");
  const [hoveredToken, setHoveredToken] = useState<TokenData | null>(null);
  const [colorMode, setColorMode] = useState<"prob" | "entropy">("prob");
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(1.0);
  const [maxNewTokens, setMaxNewTokens] = useState(30);
  const [topK, setTopK] = useState(5);
  const [colorScheme, setColorScheme] = useState<ColorScheme>("blueOrange");

  const generateTokens = async () => {
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/generate", {
        prompt,
        temperature,
        max_new_tokens: maxNewTokens,
        top_k: topK,
      });
      setTokens(response.data.tokens);
      setFullText(response.data.full_text);
      setHoveredToken(null);
    } catch (error) {
      console.error("Error generating tokens:", error);
    }
    setLoading(false);
  };

  // Compute min and max for the selected color mode
  const values = tokens.map((t) => (colorMode === "prob" ? t.prob : t.entropy));
  const minVal = values.length ? Math.min(...values) : 0;
  const maxVal = values.length ? Math.max(...values) : 1;

  return (
    <div className="p-6 min-h-screen bg-background text-foreground">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Token Visualization</h1>
        
        <div className="flex flex-col gap-4 mb-6">
          <div className="flex gap-4">
            <Input
              type="text"
              placeholder="Enter prompt..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="flex-1 text-base"
            />
            <Button 
              onClick={generateTokens}
              disabled={loading}
              className="bg-black text-white hover:bg-gray-800"
            >
              {loading ? "Generating..." : "Generate"}
            </Button>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="temperature">Temperature</Label>
              <Input
                id="temperature"
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="maxTokens">Max New Tokens</Label>
              <Input
                id="maxTokens"
                type="number"
                min={1}
                max={100}
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="topK">Top K</Label>
              <Input
                id="topK"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
          </div>
        </div>

        <div className="mb-6">
          <RadioGroup
            value={colorMode}
            onValueChange={(value: "prob" | "entropy") => setColorMode(value)}
            className="flex w-full border rounded-lg overflow-hidden"
          >
            <div className="flex-1">
              <RadioGroupItem
                value="prob"
                id="prob"
                className="peer sr-only"
              />
              <Label
                htmlFor="prob"
                className="flex flex-1 items-center justify-center p-3 cursor-pointer peer-data-[state=checked]:bg-primary peer-data-[state=checked]:text-primary-foreground hover:bg-muted/50"
              >
                Probability
              </Label>
            </div>
            <div className="flex-1">
              <RadioGroupItem
                value="entropy"
                id="entropy"
                className="peer sr-only"
              />
              <Label
                htmlFor="entropy"
                className="flex flex-1 items-center justify-center p-3 cursor-pointer peer-data-[state=checked]:bg-primary peer-data-[state=checked]:text-primary-foreground hover:bg-muted/50"
              >
                Entropy
              </Label>
            </div>
          </RadioGroup>
        </div>

        <div className="mb-6 space-y-2">
          <Label htmlFor="colorScheme">Color Scheme</Label>
          <div className="flex gap-4 items-start">
            <select
              id="colorScheme"
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value as ColorScheme)}
              className="p-2 border rounded-md bg-background"
            >
              <option value="blueOrange">Blue-Orange</option>
              <option value="greenRed">Green-Red</option>
            </select>
            
            <ColorLegend 
              scheme={colorScheme}
              min={minVal}
              max={maxVal}
              label={colorMode === "prob" ? "Probability" : "Entropy"}
              colorMode={colorMode}
            />
          </div>
        </div>

        <div className="text-2xl leading-relaxed mb-6 relative">
          {tokens.map((token, index) => {
            const value = colorMode === "prob" ? token.prob : token.entropy;
            const backgroundColor = mapValueToColor(value, minVal, maxVal, colorScheme, colorMode);
            return (
              <div
                key={index}
                className="inline-block relative group"
              >
                <span
                  className="mx-1 cursor-pointer rounded px-1"
                  style={{ 
                    backgroundColor,
                    color: 'rgba(255, 255, 255, 0.95)',
                    textShadow: '0px 0px 2px rgba(0, 0, 0, 0.3)',
                  }}
                >
                  {token.text}
                </span>
                <div className="absolute left-0 top-full mt-2 w-[400px] bg-popover text-popover-foreground p-4 rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                  <div className="w-full">
                    <h3 className="text-lg font-semibold mb-2">
                      Token: "{token.text}"
                    </h3>
                    <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                      <div>
                        <p className="font-semibold">Log Probability:</p>
                        <p>{token.log_prob.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="font-semibold">Entropy:</p>
                        <p>{token.entropy.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="font-semibold">Probability:</p>
                        <p>{token.prob.toFixed(4)}</p>
                      </div>
                    </div>
                    {(() => {
                      // Create sorted pairs of tokens and logits
                      const pairs = token.top_tokens.map((t, i) => ({
                        token: t,
                        logit: token.top_logits[i]
                      }));
                      // Sort by logits in descending order
                      pairs.sort((a, b) => b.logit - a.logit);
                      
                      return (
                        <Plot
                          data={[
                            {
                              type: "bar",
                              x: pairs.map(p => p.logit),
                              y: pairs.map(p => p.token),
                              orientation: "h",
                              marker: { color: "#1f77b4" },
                            },
                          ]}
                          layout={{
                            title: "Logits for Top Candidates",
                            xaxis: { title: "Logits" },
                            yaxis: { 
                              title: "Tokens", 
                              automargin: true,
                            },
                            margin: { l: 100, r: 20, t: 40, b: 40 },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: 'currentColor' },
                          }}
                          style={{ width: "100%", height: "300px" }}
                          config={{ responsive: true }}
                        />
                      );
                    })()}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {fullText && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Full Generated Text</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="font-mono whitespace-pre-wrap">{fullText}</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}