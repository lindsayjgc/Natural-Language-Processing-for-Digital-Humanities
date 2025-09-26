import { Card } from "../ui/card";

interface ReadabilityCardsProps {
    flesch: string,
    grade: string,
}

export function ReadabilityCards({ flesch, grade }: ReadabilityCardsProps) {

    return (
        <div className="flex flex-col md:flex-row gap-6 w-full">
            <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg">
                <h3 className="text-lg font-medium text-blue-500 mb-3">Flesch Reading Ease</h3>
                <div className="text-4xl font-bold text-gray-900 mb-2">
                    {flesch}
                </div>
                <p className="text-sm text-gray-500">Indicates how easy the text is to read. Higher scores
                    mean easier readability.</p>
            </Card>

            <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium text-blue-500 mb-3">Flesch-Kincaid Grade Level</h3>
                <div className="text-4xl font-bold text-gray-900 mb-2">
                    {grade}
                </div>
                <p className="text-sm text-gray-500">Estimates the U.S. grade level needed to understand
the text.</p>
            </Card>

        </div>
    )
}