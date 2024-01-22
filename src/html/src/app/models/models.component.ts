import {Component, OnInit} from '@angular/core';
import {PylungService} from "../pylung.service";
import {ModelList} from "../classes/model";


@Component({
  selector: 'app-models',
  templateUrl: './models.component.html',
  styleUrls: ['./models.component.scss']
})
export class ModelsComponent implements OnInit {

  modelsList: ModelList = {details_list: [], incomplete_models_list: [], error_list: []};
  name: any;
  code: any;
  constructor(private pylungService: PylungService) { }

  ngOnInit(): void {
    this.pylungService.getModels().subscribe((modelList: ModelList) => {
      console.log(modelList);
      this.modelsList = modelList;

    });
  }

  saveModel() {
    this.pylungService.saveModel(this.name, this.code).subscribe(obj => {

      if(!!obj.error) {
          this.pylungService.showNotification('Error while saving: ' + obj.error, 'danger', 'top', 'right');
      } else {
          this.name = '';
          this.code = '';
          this.ngOnInit();
          console.log(obj.filename);
          this.pylungService.showNotification('Model saved successfully', 'success', 'top', 'right');
      }
    }, error => {
        this.pylungService.showNotification('Error while saving model', 'danger', 'top','right');
    });
  }


    getPylungService() {
        return this.pylungService;
    }
}
