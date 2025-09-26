export const metadata = { title: "Stats" };

export default function StatsPage() {
    return (
        <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
            <main className="mx-auto max-w-6xl px-4 py-10">
                <h1 className="text-3xl font-semibold mb-2">Stats</h1>
                <p className="text-gray-600 mb-8">Analytics and insights from your literary texts.</p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                        <h3 className="text-lg font-medium mb-2">Total Documents</h3>
                        <p className="text-3xl font-bold text-violet-600">24</p>
                    </div>
                    <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                        <h3 className="text-lg font-medium mb-2">Words Analyzed</h3>
                        <p className="text-3xl font-bold text-violet-600">156,789</p>
                    </div>
                    <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                        <h3 className="text-lg font-medium mb-2">Genres</h3>
                        <p className="text-3xl font-bold text-violet-600">8</p>
                    </div>
                </div>
            </main>
        </div>
    );
}
