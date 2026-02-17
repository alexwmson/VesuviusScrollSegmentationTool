export interface ParamsProp{
    global_threshold: number | null;
    allowed_difference: number | null;
    max_patience: number | null;
    min_size: number | null;
    max_size: number | null;
    steps: number | null;
}

export interface FormStateProp{
    volume: string;
    seed: number[];
    params: ParamsProp;
}
