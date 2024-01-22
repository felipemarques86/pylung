import { Component, OnInit } from '@angular/core';
import {PylungService} from "../pylung.service";
import {ModelList} from "../classes/model";
import {forkJoin} from "rxjs";
import {DatasetList} from "../classes/dataset";
import {DataTransformerList} from "../classes/data-transformer-list";

@Component({
  selector: 'app-studies',
  templateUrl: './studies.component.html',
  styleUrls: ['./studies.component.scss']
})
export class StudiesComponent implements OnInit {
  batch_size =  32;
  epochs = 10;
  trainSize = 0.8;
  imageSize = 224;
  modelType: any;
  trials = 10;
  dataTransformer: any;
  dataset: any;
  database: any;
  newDatabase: any;
  noduleOnly: any;
  databases = [];
  models: ModelList;
  datasets: DatasetList;
  transformers: DataTransformerList;

  constructor(private pylungService: PylungService) { }

  ngOnInit(): void {
    this.batch_size = 32;
    this.epochs = 10;
    this.trainSize = 0.8;
    this.imageSize = 224;
    this.trials = 10;
    this.dataTransformer = '';
    this.dataset = '';
    this.newDatabase = '';
    this.database = '';
    this.noduleOnly = false;

    forkJoin(
        [this.pylungService.getDatabases(), this.pylungService.getModels(), this.pylungService.getDatasets(), this.pylungService.getDataTransformers()])
        .subscribe( (value) => {
          this.databases = value[0];
          this.models = value[1];
          this.datasets = value[2];
          this.transformers = value[3];
        }
    );
  }W

  startStudy() {
    const params = {
      'batch_size': this.batch_size,
      'epochs': this.epochs,
      'train_size': this.trainSize,
      'image_size': this.imageSize,
      'model_type': this.modelType,
      'n_trials': this.trials,
      'data_transformer_name': this.dataTransformer,
      'data_set': this.dataset,
      'db_name': !!this.newDatabase ? this.newDatabase : this.database,
      'isolate_nodule_image': this.noduleOnly ? 'True': 'False'
    };
    console.log(params)
    this.pylungService.startStudy(params).subscribe((ret) => {
      this.pylungService.showNotification('Study started successfully', 'success', 'top', 'right');
      this.ngOnInit();
    });
  }

  openOptuna(name: string) {
    alert('Starting Optuna. Please wait 5 seconds for the server to start - then a new tab will open');
     this.pylungService.startOptuna(name).subscribe((port: number) => {
        if(port > 0) {
          setTimeout( () => {
            window.open(`http://${window.location.hostname}:${port}`)
          }, 5000);
        } else {
          alert('Error while starting optuna. Check logs');
        }

     });
  }

    getPylungService() {
        return this.pylungService;
    }
}
