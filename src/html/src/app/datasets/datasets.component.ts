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
  constructor(private pylungService: PylungService) { }

  ngOnInit(): void {
    this.pylungService.getDatasets().subscribe((datasetList: DatasetList) => {
      console.log(datasetList);
      this.datasetList = datasetList;

    });
  }

    getPylungService() {
        return this.pylungService;
    }
}
