import { Component, OnInit } from '@angular/core';
import {PylungService} from "../pylung.service";
import {Dataset, DatasetList} from "../classes/dataset";
import {data} from "jquery";

@Component({
  selector: 'app-datasets',
  templateUrl: './datasets.component.html',
  styleUrls: ['./datasets.component.scss']
})
export class DatasetsComponent implements OnInit {


  datasetList: DatasetList = {directory: '', datasets: []};
  showLoader: boolean = false;
  constructor(private pylungService: PylungService) { }

  ngOnInit(): void {
    this.showLoader = true;
    this.pylungService.getDatasets().subscribe((datasetList: DatasetList) => {
      this.datasetList = datasetList;
      this.showLoader = false;
    });
  }

    getPylungService() {
        return this.pylungService;
    }
}
