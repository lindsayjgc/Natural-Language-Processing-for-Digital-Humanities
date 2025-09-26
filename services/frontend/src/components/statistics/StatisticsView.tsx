import { SummaryCards } from "./SummaryCards";
import { ReadabilityCards } from "./ReadabilityCards";
import { SectionBreak } from "./SectionBreak";
import { SentimentCards } from "./SentimentCards";
import { KeywordsCards } from "./KeywordsCards";
import { TitleSection } from "./TitleSection";

export function StatisticsView() {

    //this is all temp data, ill change it so we accept it as props or do the api call here 

    //for the pie charts in sentiment 
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

    // for the keywords card 
    const topKeywords = ['Technology', 'Innovation', 'Future', 'AI', 'Data'];

    const keywordFrequencies = [
        { keyword: 'AI', count: 12 },
        { keyword: 'Machine Learning', count: 8 },
        { keyword: 'GPT-4', count: 5 }
    ];


    return (
        <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
            <main className="mx-auto max-w-6xl px-4 py-10">
                <TitleSection/>

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

                <SectionBreak title="Keywords" />
                <KeywordsCards
                    topKeywords={topKeywords}
                    keywordFrequencies={keywordFrequencies}
                />


            </main>
        </div>
    );
}
