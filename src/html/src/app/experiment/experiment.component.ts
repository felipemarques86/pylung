import {Component, OnInit} from '@angular/core';
import {ModelList} from "../classes/model";
import {Dataset, DatasetList, Prediction, Trial} from "../classes/dataset";
import {DataTransformerList} from "../classes/data-transformer-list";
import {PylungService} from "../pylung.service";
import {forkJoin} from "rxjs";

@Component({
    selector: 'app-experiment',
    templateUrl: './experiment.component.html',
    styleUrls: ['./experiment.component.scss']
})
export class ExperimentComponent implements OnInit {

    batch_size = 32;
    epochs = 10;
    trainSize = 0.8;
    imageSize = 224;
    modelType: any;
    trials = 10;
    dataTransformer: any;
    dataset: Dataset;
    database: any;
    newDatabase: any;
    noduleOnly: any;
    databases = [];
    models: ModelList;
    datasets: DatasetList;
    transformers: DataTransformerList;
    datasetList: DatasetList;

    numbersArray: number[] = [];

    // The number of items to show per page
    itemsPerPage = 12;

    // The current page number
    currentPage = 1;

    trialList: string[] = [];

    trial: string;
    prediction: Prediction;
    selected: number;

    showLoader: boolean = false;
    selected_trial: Trial;

    get totalPages(): number[] {
        const totalPages = Math.ceil(this.numbersArray.length / this.itemsPerPage);
        const pageNumbers = [];
        for (let i = 1; i <= totalPages; i++) {
            pageNumbers.push(i);
        }
        return pageNumbers;
    }

    constructor(protected pylungService: PylungService) {
    }

    ngOnInit(): void {
        this.showLoader = true;
        this.pylungService.getDatasets().subscribe((datasetList: DatasetList) => {
            this.datasetList = datasetList;
            this.pylungService.getTrials().subscribe((trials: string[]) => {
                this.trialList = trials;
                this.showLoader = false;
            }, error => {
            console.error(error);
            this.showLoader = false;
            this.pylungService.showNotification('Error getting list of trials', 'danger', 'top','right');
             });
        }, error => {
            this.showLoader = false;
            console.error(error);
            this.pylungService.showNotification('Error getting list of data sets', 'danger', 'top','right');
        });
    }

    getDisplayedItems(starting_from: string): number[] {
        let starting_fromNr = !starting_from ? 0 : Number(starting_from);
        const startIndex = (this.currentPage - 1) * this.itemsPerPage + starting_fromNr;
        const endIndex = startIndex + this.itemsPerPage;
        console.log(startIndex, endIndex);
        return this.numbersArray.slice(startIndex, endIndex);
    }

    // A method to change the current page

    changePage(pageNumber: number): void {
        this.currentPage = pageNumber;
    }

    open(row: Dataset) {
        this.dataset = row;
        console.log(row);
        this.numbersArray = [];
        for (let i = 0; i <= this.dataset.count; i++) {
            this.numbersArray.push(i);
        }
    }

    predict(i: number) {
        this.selected = i;
        this.showLoader = true;
        this.pylungService.predict(this.trial, this.dataset.type, this.dataset.name, i).subscribe(prediction => {
            this.prediction = prediction;
            this.showLoader = false;
        }, error => {
            this.showLoader = false;
            console.error(error);
            this.pylungService.showNotification('Error running prediction model', 'danger', 'top','right');
        });
    }

    loadTrial(trial_name: string) {
        this.showLoader = true;
        this.pylungService.getTrialDetails(trial_name).subscribe((_trial: Trial) => {
            this.selected_trial = _trial
            this.showLoader = false;
        }, error => {
            this.showLoader = false;
            console.error(error);
            this.pylungService.showNotification('Error loading trial', 'danger', 'top','right');
        });
    }
}
