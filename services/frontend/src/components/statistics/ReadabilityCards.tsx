import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface ReadabilityCardsProps {
  flesch: string;
  grade: string;
}

export function ReadabilityCards({ flesch, grade }: ReadabilityCardsProps) {
  return (
    <div className="flex flex-col md:flex-row gap-6 w-full">
      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Flesch Reading Ease
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold text-gray-900 mb-2">{flesch}</div>
          <CardDescription>
            Indicates how easy the text is to read. Higher scores mean easier
            readability.
          </CardDescription>
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-lg font-medium text-blue-500">
            Flesch-Kincaid Grade Level
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold text-gray-900 mb-2">{grade}</div>
          <CardDescription>
            Estimates the U.S. grade level needed to understand the text.
          </CardDescription>
        </CardContent>
      </Card>
    </div>
  );
}
