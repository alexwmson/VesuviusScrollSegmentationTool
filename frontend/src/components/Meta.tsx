import type { MetaProps } from "../interfaces/MetaProps"

const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function Meta({meta, uuid}: MetaProps){
    async function exportSegment(){
        if (!meta) return;
        try{
            const res = await fetch(`${API_BASE}/segments/${uuid}/download`);
            const blob = await res.blob();

            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `segment-${uuid}.zip`;
            a.click();
            URL.revokeObjectURL(url);
        }
        catch(e){
            console.error("Failed to export segment", e);
        }
    }

    function prettyDate(ts: string): string {
        const year = Number(ts.slice(0, 4));
        const month = Number(ts.slice(4, 6)) - 1;
        const day = Number(ts.slice(6, 8));
        const hour = Number(ts.slice(8, 10));
        const minute = Number(ts.slice(10, 12));
        const second = Number(ts.slice(12, 14));
        const ms = Number(ts.slice(14, 17));
        const date = new Date(year, month, day, hour, minute, second, ms);

        return date.toLocaleString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    }

    return (
        <div className="metaDiv">
            <button onClick={() => exportSegment()} disabled={meta ? false : true}>Export Segment</button>
            <div>
                <div>Meta Data</div>
                {meta && 
                    <div className="details">
                        <div className="metaData">Starting point (x,y,z): {meta.starting_points[0]}, {meta.starting_points[1]}, {meta.starting_points[2]}</div>
                        <div className="metaData">Voxels explored: {meta.voxels_explored}</div>
                        <div className="metaData">Pixels in grayscale: {meta.pixels_on_grayscale}</div>
                        <div className="metaData">Created on: {prettyDate(meta.created_on)}</div>
                    </div>
                }
                {!meta &&
                    <div className="details">
                        <div>Data will be displayed here upon viewing a segment</div>
                    </div>
                }
            </div>
        </div>
    );
}