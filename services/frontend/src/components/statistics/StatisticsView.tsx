import { SummaryCards } from "./SummaryCards";
import { ReadabilityCards } from "./ReadabilityCards";
import { SectionBreak } from "./SectionBreak";
import { SentimentCards } from "./SentimentCards";

export function StatisticsView() {

    //this is temp data for the pie charts in sentiment 
    const polarityData = [
        { name: 'Positive', value: 75, color: '#10B981' },
        { name: 'Neutral', value: 15, color: '#D1D5DB' },
        { name: 'Negative', value: 10, color: '#EF4444' }
    ];

    const analysisData = [
        { name: 'Joy', value: 60, color: '#10B981' },
        { name: 'Surprise', value: 20, color: '#F59E0B' },
        { name: 'Sadness', value: 15, color: '#3B82F6' },
        { name: 'Anger', value: 5, color: '#EF4444' }
    ];


    return (
        <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
            <main className="mx-auto max-w-6xl px-4 py-10">
                <h1 className="text-3xl font-semibold mb-2">Statistics Dashboard</h1>
                <h2 className="text-xl text-gray-600">Name of Document</h2>

                <SectionBreak title="Summary" />
                <SummaryCards
                    characters={"7,542"}
                    words={"1,250"}
                    sentences={"75"}
                />

                <SectionBreak title="Readability" />
                <ReadabilityCards
                    flesch={"68"}
                    grade={"8"}
                />

                <SectionBreak title="Sentiment" />
                <SentimentCards
                    polarityData={polarityData}
                    analysisData={analysisData}
                />


            </main>
        </div>
    );
}
