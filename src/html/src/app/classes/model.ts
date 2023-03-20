export interface Model {
    model_name: string;
    parameters: string;
    description: string;
    extra_information: string;
}

export interface ModelError {
    model: string;
    error: string;
}

export interface ModelList {
    details_list: Model[];
    incomplete_models_list: string[];
    error_list: ModelError[];
}
