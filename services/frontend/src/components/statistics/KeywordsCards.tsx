import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface KeywordFrequency {
  keyword: string;
  count: number;
}

interface KeywordsCardsProps {
  topKeywords: string[];
  keywordFrequencies: KeywordFrequency[];
}

export function KeywordsCards({
  topKeywords,
  keywordFrequencies,
}: KeywordsCardsProps) {
  const maxCount =
    keywordFrequencies.length > 0
      ? Math.max(...keywordFrequencies.map((item) => item.count))
      : 1;

  if (topKeywords.length === 0 && keywordFrequencies.length === 0) {
    return (
      <div className="flex flex-col md:flex-row gap-6 w-full">
        <Card className="flex-1">
          <CardHeader>
            <CardTitle className="text-lg font-medium text-blue-500">
              Keywords
            </CardTitle>
          </CardHeader>
          <CardContent>
            <CardDescription>No keyword data available.</CardDescription>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row gap-6 w-full">
      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Top Keywords
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {topKeywords.length > 0 ? (
              topKeywords.map((keyword) => (
                <Badge
                  key={keyword}
                  variant="secondary"
                  className="bg-blue-100 text-blue-400"
                >
                  {keyword}
                </Badge>
              ))
            ) : (
              <CardDescription>No keywords available.</CardDescription>
            )}
          </div>
          <CardDescription>Most frequently occurring keywords.</CardDescription>
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Keyword Frequency
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {keywordFrequencies.length > 0 ? (
              keywordFrequencies.map((item) => (
                <div key={item.keyword} className="flex items-center gap-4">
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
              ))
            ) : (
              <CardDescription>No frequency data available.</CardDescription>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
