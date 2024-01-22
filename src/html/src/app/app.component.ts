import { Component, OnInit } from '@angular/core';
import { LocationStrategy, PlatformLocation, Location } from '@angular/common';
import {PylungService, UI} from "./pylung.service";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
     constructor(public location: Location, public pylungService:PylungService) {}

    ngOnInit(){
         this.pylungService.loggedIn = false;
         this.pylungService.getUi().subscribe(ui => {
            this.pylungService.ui = ui;
         });
    }

    isMap(path){
      var titlee = this.location.prepareExternalUrl(this.location.path());
      titlee = titlee.slice( 1 );
      if(path == titlee){
        return false;
      }
      else {
        return true;
      }
    }
}
