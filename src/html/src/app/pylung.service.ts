import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Dataset, DatasetList, Prediction, Trial} from "./classes/dataset";
import {Observable} from "rxjs";
import {ModelList} from "./classes/model";
import {DataTransformerList} from "./classes/data-transformer-list";

declare var $:any;

export interface UI {
    'public': boolean;
    'ui': {
        'visibility': {
            'Models': boolean,
            'Models.Models_List': boolean,
            'Models.Incomplete_Models': boolean,
            'Models.Models_with_errors': boolean,
            'Models.Register_New_Model': boolean,
            'Datasets': boolean,
            'Datasets.Dataset_List': boolean,
            'Studies': boolean,
            'Studies.Database_List': boolean,
            'Studies.Create_New_Study': boolean,
            'Experiments': boolean,
            'Experiments.Dataset_List': boolean,
            'Experiments.Trial_list': boolean,
            'Experiments.Image_List': boolean,
            'Experiments.Result_of_Image': boolean
        }
    }
}

@Injectable({
  providedIn: 'root'
})
export class PylungService {
    ui: UI;
    loggedIn: boolean;

  constructor(private http: HttpClient) { }

  getDatasets(): Observable<DatasetList>{
    const url = '/rest/datasets/lidc_idri';
    return this.http.get<DatasetList>(url);
  }

  getModels(): Observable<ModelList>{
    const url = '/rest/models';
    return this.http.get<ModelList>(url);
  }

  getDataTransformers(): Observable<DataTransformerList> {
      const url = '/rest/datatransformers'
      return this.http.get<DataTransformerList>(url);
  }

  saveModel(name: any, code: any) {
      const url = '/rest/models';
      return this.http.post<any>(url, {name: name, code: code});
  }

    startStudy(params: {}) {
    const url = '/rest/studies';
      return this.http.post<any>(url, params);
    }

    showNotification(message, type, from, align){

      $.notify({
          icon: "pe-7s-bell",
          message: message
      },{
          type: type,
          timer: 1000,
          placement: {
              from: from,
              align: align
          }
      });
  }

    getDatabases() {
        const url = '/rest/databases';
        return this.http.get<string[]>(url);
    }

    startOptuna(name: string) {
        const url = `/rest/optuna/start/${name}.sqlite3`;
        return this.http.get<number>(url);
    }

    getImageIndex(i: number, r: Dataset) {
      if (!!r.starting_from) {
          i = i + Number(r.starting_from);
      }
      return i;
    }

    getImageUrl(r: Dataset, i: number, data = '', bbox = false, crop = false, ignore_starting_from = false) {
      if (!!r.starting_from && !ignore_starting_from) {
          i = i + Number(r.starting_from);
      }
      if (bbox)
        return `/rest/navigate/${r.type}/${r.name}/image-${i}.png?bbox=True&data=${data}`;
      else if (crop)
        return `/rest/navigate/${r.type}/${r.name}/image-${i}.png?crop=True&data=${data}`;
      else
          return `/rest/navigate/${r.type}/${r.name}/image-${i}.png`;
    }

    predict(trial, ds_type, ds_name, index) {
        const url = `/rest/predict/${trial}/${ds_type}/${ds_name}/${index}`;
        return this.http.get<Prediction>(url);
    }

    getTrials() {
      const url = '/rest/trials';
      return this.http.get<string[]>(url);
    }

    getTrialDetails(trial_name: string) {
      const url = `/rest/trials/${trial_name}`;
      return this.http.get<Trial>(url);
    }

    getUi() {
        const url = '/rest/ui';
        return this.http.get<UI>(url);
    }
}
