import React, { useState } from "react";
import type { FormProps } from "../interfaces/FormProps"
import type { FormStateProp } from "../interfaces/FormStateProp";
import type { ParamsProp } from "../interfaces/FormStateProp";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function Form({volume, setSegments, seed, setSeed, isFocusPoint, setIsFocusPoint} : FormProps){
    const { x, y, z } = seed;
    const [params, setParams] = useState<ParamsProp>({
        global_threshold: 112,
        allowed_difference: 0.95,
        max_patience: 5,
        min_size: 50000,
        max_size: 250000,
        steps: 10
    });

    function coordChange(e: React.ChangeEvent<HTMLInputElement>){
        let value  = e.currentTarget.value.trim() === "" ? null : +e.currentTarget.value;
        const name = e.target.name as "x" | "y" | "z";

        setSeed((prev: { x: number | null; y: number | null; z: number | null }) => ({ ...prev, [name]: value }));
    }

    function paramsChange(e: React.ChangeEvent<HTMLInputElement>){
        const { name, value } = e.currentTarget;

        if (value == "")
            setParams(prev => ({
                ...prev,
                [name] : null
            }))

        else
            setParams(prev => ({
                ...prev,
                [name] : +value
            }))
    }

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>){
        e.preventDefault();
        const seed: number[] = (x !== null && y !== null && z !== null) ? [x, y, z]: [];
        const finalParams: Partial<ParamsProp> = {};
        (Object.keys(params) as (keyof ParamsProp)[]).forEach(key =>{
            if (params[key] !== null){
                finalParams[key] = params[key];
            }
        })

        const completeJson = {
            volume: volume,
            seed: seed,
            params: finalParams
        }

        console.log("Sending params:", completeJson);

        try {
            const response = await fetch(`${API_BASE}/generate_segment`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(completeJson)
            });

            //console.log(JSON.stringify(completeJson))
            if (!response.ok)
                throw new Error(`Request failed with status ${response.status}`);

            const data = await response.json();
            console.log("Segmentation task started:", data);
            
            setSegments((prev: [string, number][]) => [...prev, [data.uuid, 0]]);
            pingJobStatus(data.uuid);
        }
        catch(error) {
            console.error("Failed to start segmentation:", error);
        }

        return;
    }

    function pingJobStatus(uuid: string) {
        const interval = setInterval(async () => {
            try {
            const res = await fetch(`${API_BASE}/jobs/${uuid}`);
            const status = await res.json();

            console.log("Job status:", status);

            if (status.state === "SUCCESS") {
                setSegments((prev: [string, number][]) => prev.map(([key, value]) => uuid === key ? [key, 1] : [key, value]));
                clearInterval(interval);
                console.log("Job finished:", status.result);
            }

            if (status.state === "FAILURE") {
                setSegments((prev: [string, number][]) => prev.filter(([key]) => key !== uuid));
                clearInterval(interval);
                console.error("Job failed:", status.error);
            }

            } catch (err) {
                setSegments((prev: [string, number][]) => prev.filter(([key]) => key !== uuid));
                clearInterval(interval);
                console.error("Polling failed:", err);
            }
        }, 2000);
    }

    return (
        <form onSubmit={handleSubmit}>
            <label>
                Seed
                <div className="seedRow">
                    <label>
                        x: 
                        <input name="x" type="number" onChange={coordChange} value={x !== null ? x : ""}/>
                    </label>
                    <label>
                        y: 
                        <input name="y" type="number" onChange={coordChange} value={y !== null ? y : ""}/>
                    </label>
                    <label>
                        z: 
                        <input name="z" type="number" onChange={coordChange} value={z !== null ? z : ""}/>
                    </label>
                    <button type="button" onClick={() => {
                        setIsFocusPoint(false);
                        setSeed({x: null, y: null, z: null});
                        }}>Reset focus point</button>
                </div>
            </label>
            <label>
                Additional Parameters
                <div>
                    <label>
                        Min Size: 
                        <input name="min_size" type="number" min={1} max={1_000_000} onChange={paramsChange} value={params.min_size !== null ? params.min_size : ""}/>
                    </label>
                    <label>
                        Max Size: 
                        <input name="max_size" type="number" min={1} max={1_000_000} onChange={paramsChange} value={params.max_size !== null ? params.max_size : ""}/>
                    </label>
                    <label>
                        Step Size: 
                        <input name="steps" type="number" min={1} max={40} onChange={paramsChange} value={params.steps !== null ? params.steps : ""}/>
                    </label>
                </div>
                <div>
                    <label>
                        Global Threshold:
                        <input name="global_threshold" type="number" min={0} max={255} onChange={paramsChange} value={params.global_threshold !== null ? params.global_threshold : ""} />
                    </label>
                    <label>
                        Allowed Difference:
                        <input name="allowed_difference" type="number" min={-1} max={1} step={0.01} onChange={paramsChange} value={params.allowed_difference !== null ? params.allowed_difference : ""} />
                    </label>
                    <label>
                        Patience Max:
                        <input name="max_patience" type="number" min={1} max={50} onChange={paramsChange} value={params.max_patience !== null ? params.max_patience : ""} />
                    </label>
                </div>
            </label>
            <button type="submit">Generate Segment</button>
        </form>
    );
}