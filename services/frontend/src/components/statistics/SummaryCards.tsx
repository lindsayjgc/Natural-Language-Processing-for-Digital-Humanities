import { Card } from "../ui/card";

interface SummaryCardsProps {
    characters: string,
    words: string,
    sentences: string,
}

export function SummaryCards({ characters, words, sentences }: SummaryCardsProps) {

    return (
        <div className="flex flex-col md:flex-row gap-6 w-full">
            <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg">
                <h3 className="text-lg font-medium text-blue-500 mb-3">Character Count</h3>
                <div className="text-4xl font-bold text-gray-900 mb-2">
                    {characters}
                </div>
                <p className="text-sm text-gray-500">Total number of characters.</p>
            </Card>

            <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium text-blue-500 mb-3">Word Count</h3>
                <div className="text-4xl font-bold text-gray-900 mb-2">
                    {words}
                </div>
                <p className="text-sm text-gray-500">Total number of words.</p>
            </Card>

            <Card className="flex-1 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium text-blue-500 mb-3">Sentence Count</h3>
                <div className="text-4xl font-bold text-gray-900 mb-2">
                    {sentences}
                </div>
                <p className="text-sm text-gray-500">Total number of sentences.</p>
            </Card>
            
        </div>
    )
}