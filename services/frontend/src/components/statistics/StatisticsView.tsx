import { SummaryCards } from "./SummaryCards";

export function StatisticsView() {
    return (
        <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
            <main className="mx-auto max-w-6xl px-4 py-10">
                <h1 className="text-3xl font-semibold mb-2">Statistics Dashboard</h1>
                <h2 className="text-xl text-gray-600 mb-5">Name of Document</h2>

                <h1 className="text-2xl font-semibold mb-2">Summary</h1>
                <hr className="h-px my-4 mb-8 bg-gray-200 border-0 dark:bg-gray-700"/>

                <SummaryCards
                    characters={"7,542"}
                    words={"1,250"}
                    sentences={"75"}
                />
            </main>
        </div>
    );
}
