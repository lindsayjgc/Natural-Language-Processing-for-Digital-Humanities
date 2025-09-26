import { Card } from "@/components/ui/card";

interface KeywordFrequency {
  keyword: string;
  count: number;
}

interface KeywordsCardsProps {
  topKeywords: string[];
  keywordFrequencies: KeywordFrequency[];
}

export function KeywordsCards({ topKeywords, keywordFrequencies }: KeywordsCardsProps) {
  const maxCount = Math.max(...keywordFrequencies.map(item => item.count));

  return (
    <div className="flex flex-col md:flex-row gap-6 w-full">
      <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg">
        <h3 className="text-lg font-medium text-blue-500 mb-4">Top Keywords</h3>
        <div className="flex flex-wrap gap-2 mb-4">
          {topKeywords.map((keyword, index) => (
            <span
              key={index}
              className="px-3 py-1 bg-blue-100 text-blue-400 rounded-full text-sm font-medium"
            >
              {keyword}
            </span>
          ))}
        </div>
        <p className="text-sm text-gray-500">Most frequently occurring keywords.</p>
      </Card>

      <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg">
        <h3 className="text-lg font-medium text-blue-500 mb-4">Keyword Frequency</h3>
        <div className="space-y-4">
          {keywordFrequencies.map((item, index) => (
            <div key={index} className="flex items-center gap-4">
              <div className="w-20 text-sm font-medium text-gray-700 text-right">
                {item.keyword}
              </div>
              <div className="flex-1 flex items-center gap-2">
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${(item.count / maxCount) * 100}%` }}
                  />
                </div>
                <div className="w-6 text-sm font-medium text-gray-600 text-right">
                  {item.count}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}