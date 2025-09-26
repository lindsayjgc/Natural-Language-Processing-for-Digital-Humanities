'use client';
import { Card } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface SentimentData {
  name: string;
  value: number;
  color: string;
  [key: string]: any; 
}

interface SentimentCardsProps {
  polarityData: SentimentData[];
  analysisData: SentimentData[];
}

const Legend = ({ data }: { data: SentimentData[] }) => (
  <div className="flex flex-col gap-2">
    {data.map((item, index) => (
      <div key={index} className="flex items-center gap-2">
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

export function SentimentCards({ polarityData, analysisData }: SentimentCardsProps) {
  return (
    <div className="flex flex-col md:flex-row gap-6 w-full">
      <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
        <h3 className="text-lg font-medium text-blue-500 mb-6">Sentiment Polarity</h3>
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
                  {polarityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <Legend data={polarityData} />
        </div>
      </Card>

      <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
        <h3 className="text-lg font-medium text-blue-500 mb-6">Sentiment Analysis</h3>
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
                  {analysisData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <Legend data={analysisData} />
        </div>
      </Card>
    </div>
  );
}