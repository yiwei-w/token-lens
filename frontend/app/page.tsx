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

function mapValueToColor(value: number, min: number, max: number): string {
  const norm = max > min ? (value - min) / (max - min) : 0.5;
  const r = Math.round(255 * (1 - norm));
  const b = Math.round(255 * norm);
  return `rgb(${r}, 100, ${b})`;
}

export default function Page() {
  const [prompt, setPrompt] = useState("");
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [fullText, setFullText] = useState("");
  const [hoveredToken, setHoveredToken] = useState<TokenData | null>(null);
  const [colorMode, setColorMode] = useState<"prob" | "entropy">("prob");
  const [loading, setLoading] = useState(false);

  const generateTokens = async () => {
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/generate", {
        prompt,
        temperature: 1.0,
        max_new_tokens: 30,
        top_k: 5,
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
      <h1 className="text-2xl font-bold mb-6">Token Visualization</h1>
      
      <div className="flex gap-4 mb-6">
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
        >
          {loading ? "Generating..." : "Generate"}
        </Button>
      </div>

      <RadioGroup
        value={colorMode}
        onValueChange={(value: "prob" | "entropy") => setColorMode(value)}
        className="flex gap-4 mb-6"
      >
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="prob" id="prob" />
          <Label htmlFor="prob">Probability</Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="entropy" id="entropy" />
          <Label htmlFor="entropy">Entropy</Label>
        </div>
      </RadioGroup>

      <div className="text-2xl leading-relaxed mb-6 relative">
        {tokens.map((token, index) => {
          const value = colorMode === "prob" ? token.prob : token.entropy;
          const color = mapValueToColor(value, minVal, maxVal);
          return (
            <div
              key={index}
              className="inline-block relative group"
            >
              <span
                className="mx-1 cursor-pointer"
                style={{ color }}
              >
                {token.text}
              </span>
              <div className="absolute left-0 top-full mt-2 w-[400px] bg-popover text-popover-foreground p-4 rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                <div className="w-full">
                  <h3 className="text-lg font-semibold mb-2">
                    Top Candidates for "{token.text}"
                  </h3>
                  <Plot
                    data={[
                      {
                        type: "bar",
                        x: token.top_logits,
                        y: token.top_tokens,
                        orientation: "h",
                        marker: { color: "#1f77b4" },
                      },
                    ]}
                    layout={{
                      title: "Logits for Top Candidates",
                      xaxis: { title: "Logits" },
                      yaxis: { title: "Tokens", automargin: true },
                      margin: { l: 100, r: 20, t: 40, b: 40 },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      font: { color: 'currentColor' },
                    }}
                    style={{ width: "100%", height: "300px" }}
                    config={{ responsive: true }}
                  />
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
            <p>{fullText}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}