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
  top_logprobs: number[];
  top_logits: number[];
};

type ColorScheme = "blueOrange" | "greenRed";

function mapValueToColor(
  value: number, 
  min: number, 
  max: number, 
  scheme: ColorScheme,
  colorMode: "prob" | "logprob" | "entropy"
): string {
  const norm = max > min ? (value - min) / (max - min) : 0.5;
  
  // For logprob, we want higher values (closer to 0) to be "better"
  const adjustedNorm = colorMode === "logprob" ? (1 - norm) : 
                       colorMode === "prob" ? (1 - norm) : norm;
  
  switch (scheme) {
    case "blueOrange":
      const r = Math.round(255 * adjustedNorm);
      const b = Math.round(255 * (1 - adjustedNorm));
      return `rgb(${r}, ${Math.round(r * 0.65)}, ${b})`;
    
    case "greenRed":
      const g = Math.round(255 * (1 - adjustedNorm));
      const r2 = Math.round(255 * adjustedNorm);
      return `rgb(${r2}, ${g}, 0)`;
      
    default:
      return `rgb(0, 0, 0)`;
  }
}

function generateColorScale(scheme: ColorScheme, colorMode: "prob" | "logprob" | "entropy"): string[] {
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
  colorMode: "prob" | "logprob" | "entropy";
}) {
  const colors = generateColorScale(scheme, colorMode);
  
  return (
    <div className="flex flex-col w-full">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium">{label}</span>
      </div>
      <div className="flex flex-col">
        <div className="relative h-[24px] rounded-md overflow-hidden">
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
        <div className="flex justify-between mt-1">
          <span className="text-xs font-medium">
            {min.toFixed(2)}
          </span>
          <span className="text-xs font-medium">
            {max.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

function formatWhitespaceToken(text: string): string {
  if (text === '\n') return '\\n';
  if (text === '\t') return '\\t';
  if (text === ' ') return '␣'; // Space character
  if (text === '\r') return '\\r';
  return text;
}

export default function Page() {
  const [prompt, setPrompt] = useState("");
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [fullText, setFullText] = useState("");
  const [selectedToken, setSelectedToken] = useState<TokenData | null>(null);
  const [colorMode, setColorMode] = useState<"prob" | "logprob" | "entropy">("prob");
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
      setSelectedToken(null);
    } catch (error) {
      console.error("Error generating tokens:", error);
      if (axios.isAxiosError(error)) {
        console.error("Axios error details:", error.message, error.response?.data);
      }
    }
    setLoading(false);
  };

  const copyFullText = () => {
    const textToCopy = `${prompt}${fullText}`;
    navigator.clipboard.writeText(textToCopy)
      .then(() => {
        // You could add a toast notification here if you want
        console.log('Text copied to clipboard');
      })
      .catch(err => {
        console.error('Failed to copy text: ', err);
      });
  };

  const sendToInput = () => {
    setPrompt(`${prompt}${fullText}`);
  };

  // Compute min and max for the selected color mode
  const values = tokens.map((t) => {
    if (colorMode === "prob") return t.prob;
    if (colorMode === "logprob") return t.log_prob;
    return t.entropy;
  });
  const minVal = values.length ? Math.min(...values) : 0;
  const maxVal = values.length ? Math.max(...values) : 1;

  return (
    <div className="p-6 min-h-screen bg-background text-foreground">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Token Visualization</h1>
        
        <div className="flex flex-col gap-4 mb-6">
          <div className="flex flex-col gap-2">
            <Label htmlFor="promptInput">Prompt</Label>
            <textarea
              id="promptInput"
              placeholder="Enter prompt... (use \n for line breaks)"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="flex-1 text-base p-3 min-h-[100px] resize-y border rounded-md bg-background"
              onKeyDown={(e) => {
                // Prevent actual newlines, user needs to type \n manually
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                }
              }}
            />
          </div>
          
          <Button 
            onClick={generateTokens}
            disabled={loading}
            className="bg-black text-white hover:bg-gray-800 w-full md:w-auto md:self-end"
          >
            {loading ? "Generating..." : "Generate"}
          </Button>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
              <Label htmlFor="topK">Top K Logprobs</Label>
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
            onValueChange={(value: "prob" | "logprob" | "entropy") => setColorMode(value)}
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
                value="logprob"
                id="logprob"
                className="peer sr-only"
              />
              <Label
                htmlFor="logprob"
                className="flex flex-1 items-center justify-center p-3 cursor-pointer peer-data-[state=checked]:bg-primary peer-data-[state=checked]:text-primary-foreground hover:bg-muted/50"
              >
                Log Probability
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
          <Label htmlFor="colorScheme">Color Map</Label>
          <div className="flex gap-4 items-start">
            <select
              id="colorScheme"
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value as ColorScheme)}
              className="p-2 border rounded-md bg-background h-10 mt-1"
            >
              <option value="blueOrange">Blue-Orange</option>
              <option value="greenRed">Green-Red</option>
            </select>
            
            <div className="w-48">
              <ColorLegend 
                scheme={colorScheme}
                min={minVal}
                max={maxVal}
                label={colorMode === "prob" ? "Probability" : colorMode === "logprob" ? "Log Probability" : "Entropy"}
                colorMode={colorMode}
              />
            </div>
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/2">
            <div className="text-2xl leading-relaxed mb-6 relative font-mono">
              {tokens.map((token, index) => {
                const value = colorMode === "prob" ? token.prob : colorMode === "logprob" ? token.log_prob : token.entropy;
                const backgroundColor = mapValueToColor(value, minVal, maxVal, colorScheme, colorMode);
                
                // Format whitespace characters for display
                let displayText = token.text;
                if (displayText === '\n') displayText = '\\n';
                else if (displayText === '\t') displayText = '\\t';
                else if (displayText === ' ') displayText = '␣'; // Space character
                else if (displayText === '\r') displayText = '\\r';
                
                return (
                  <div
                    key={index}
                    className="inline-block relative"
                  >
                    <span
                      className="mx-1 cursor-pointer rounded px-1 text-sm"
                      style={{ 
                        backgroundColor,
                        color: 'rgba(255, 255, 255, 0.95)',
                        textShadow: '0px 0px 2px rgba(0, 0, 0, 0.3)',
                        fontSize: '1.1rem',
                      }}
                      onClick={() => setSelectedToken(selectedToken === token ? null : token)}
                    >
                      {displayText}
                    </span>
                  </div>
                );
              })}
            </div>

            {fullText && (
              <Card className="mt-6">
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle>Full Generated Text</CardTitle>
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={sendToInput}
                      className="flex items-center gap-1"
                    >
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        width="16" 
                        height="16" 
                        viewBox="0 0 24 24" 
                        fill="none" 
                        stroke="currentColor" 
                        strokeWidth="2" 
                        strokeLinecap="round" 
                        strokeLinejoin="round"
                      >
                        <path d="M5 12h14"></path>
                        <path d="M12 5v14"></path>
                      </svg>
                      Continue from this
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={copyFullText}
                      className="flex items-center gap-1"
                    >
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        width="16" 
                        height="16" 
                        viewBox="0 0 24 24" 
                        fill="none" 
                        stroke="currentColor" 
                        strokeWidth="2" 
                        strokeLinecap="round" 
                        strokeLinejoin="round"
                      >
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>
                      Copy full text
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="mb-2 text-sm text-muted-foreground">
                    <span className="font-semibold">Prompt:</span> {prompt}
                  </div>
                  <p className="font-mono whitespace-pre-wrap">{fullText}</p>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="md:w-1/2">
            {selectedToken ? (
              <Card className="sticky top-6">
                <CardHeader>
                  <CardTitle>Token: "{formatWhitespaceToken(selectedToken.text)}"</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
                    <div>
                      <p className="font-semibold">Log Probability:</p>
                      <p>{selectedToken.log_prob.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="font-semibold">Entropy:</p>
                      <p>{selectedToken.entropy.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="font-semibold">Probability:</p>
                      <p>{selectedToken.prob.toFixed(4)}</p>
                    </div>
                  </div>
                  {(() => {
                    // Create sorted pairs of tokens with logprobs and logits
                    const pairs = selectedToken.top_tokens.map((t, i) => ({
                      token: formatWhitespaceToken(t),
                      logprob: selectedToken.top_logprobs ? selectedToken.top_logprobs[i] : 0,
                      prob: Math.exp(selectedToken.top_logprobs ? selectedToken.top_logprobs[i] : 0),
                      logit: selectedToken.top_logits ? selectedToken.top_logits[i] : 0
                    }));
                    // Sort by logprobs in descending order
                    pairs.sort((a, b) => b.logprob - a.logprob);
                    
                    return (
                      <div>
                        <Plot
                          data={[
                            {
                              type: "bar",
                              x: pairs.map(p => p.prob),
                              y: pairs.map(p => p.token),
                              orientation: "h",
                              marker: { color: "#1f77b4" },
                              name: "Probability"
                            },
                          ]}
                          layout={{
                            title: "Probabilities for Top Candidates",
                            xaxis: { 
                              title: "Probability",
                              autorange: true,
                            },
                            yaxis: { 
                              title: "Tokens", 
                              automargin: true,
                              ticktext: pairs.map(p => p.token),
                              tickvals: pairs.map((_, i) => i),
                              tickmode: "array",
                              tickalign: "left",
                              side: "left",
                              showticklabels: true,
                            },
                            margin: { l: 100, r: 20, t: 40, b: 40 },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: 'currentColor' },
                          }}
                          style={{ width: "100%", height: "300px" }}
                          config={{ responsive: true }}
                        />
                        
                        <div className="mt-4">
                          <h4 className="text-md font-semibold mb-2">Top Tokens Details</h4>
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                              <thead>
                                <tr className="border-b">
                                  <th className="text-left py-2">Token</th>
                                  <th className="text-right py-2">Probability</th>
                                  <th className="text-right py-2">Log Prob</th>
                                  <th className="text-right py-2">Logit</th>
                                </tr>
                              </thead>
                              <tbody>
                                {pairs.map((pair, idx) => (
                                  <tr key={idx} className="border-b border-gray-100">
                                    <td className="py-1 font-mono">{pair.token}</td>
                                    <td className="text-right py-1">{pair.prob.toFixed(4)}</td>
                                    <td className="text-right py-1">{pair.logprob.toFixed(4)}</td>
                                    <td className="text-right py-1">{pair.logit.toFixed(4)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </CardContent>
              </Card>
            ) : (
              <div className="h-full flex items-center justify-center p-8 border border-dashed rounded-lg">
                <p className="text-muted-foreground text-center">
                  Click on a token to view detailed statistics
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}