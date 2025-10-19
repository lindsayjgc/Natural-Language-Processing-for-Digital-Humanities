import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface SummaryCardsProps {
  characters: string;
  words: string;
  sentences: string;
}

export function SummaryCards({
  characters,
  words,
  sentences,
}: SummaryCardsProps) {
  return (
    <div className="flex flex-col md:flex-row gap-6 w-full">
      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Character Count
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold text-gray-900 mb-2">
            {characters}
          </div>
          <CardDescription>Total number of characters.</CardDescription>
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Word Count
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold text-gray-900 mb-2">{words}</div>
          <CardDescription>Total number of words.</CardDescription>
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Sentence Count
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold text-gray-900 mb-2">
            {sentences}
          </div>
          <CardDescription>Total number of sentences.</CardDescription>
        </CardContent>
      </Card>
    </div>
  );
}
