import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Dataset, DatasetList, Prediction, Trial} from "./classes/dataset";
import {Observable} from "rxjs";
import {ModelList} from "./classes/model";
import {DataTransformerList} from "./classes/data-transformer-list";

declare var $:any;
@Injectable({
  providedIn: 'root'
})
export class PylungService {

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

    getImageUrl(r: Dataset, i: number, data = '', bbox = false, crop = false) {
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
}
