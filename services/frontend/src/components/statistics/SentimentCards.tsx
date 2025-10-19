"use client";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SentimentData {
  name: string;
  value: number;
  color: string;
  [key: string]: unknown;
}

interface SentimentCardsProps {
  polarityData: SentimentData[];
  analysisData: SentimentData[];
}

const Legend = ({ data }: { data: SentimentData[] }) => (
  <div className="flex flex-col gap-2">
    {data.map((item) => (
      <div key={item.name} className="flex items-center gap-2">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: item.color }}
        />
        <span className="text-sm text-gray-600">
          {item.name} ({item.value}%)
        </span>
      </div>
    ))}
  </div>
);

export function SentimentCards({
  polarityData,
  analysisData,
}: SentimentCardsProps) {
  return (
    <div className="flex flex-col md:flex-row gap-6 w-full">
      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Sentiment Polarity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-8">
            <div className="w-32 h-32">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={polarityData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={60}
                    paddingAngle={2}
                    dataKey="value"
                    strokeWidth={0}
                  >
                    {polarityData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <Legend data={polarityData} />
          </div>
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Sentiment Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-8">
            <div className="w-32 h-32">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={analysisData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={60}
                    paddingAngle={2}
                    dataKey="value"
                    strokeWidth={0}
                  >
                    {analysisData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <Legend data={analysisData} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
