import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {HttpClientModule} from '@angular/common/http';
import {RouterModule} from '@angular/router';

import {AppRoutingModule} from './app.routing';
import {NavbarModule} from './shared/navbar/navbar.module';
import {FooterModule} from './shared/footer/footer.module';
import {SidebarModule} from './sidebar/sidebar.module';

import {AppComponent} from './app.component';

import {AdminLayoutComponent} from './layouts/admin-layout/admin-layout.component';
import { ModelsComponent } from './models/models.component';
import { StudiesComponent } from './studies/studies.component';
import { ExperimentComponent } from './experiment/experiment.component';

@NgModule({
  imports: [
    BrowserAnimationsModule,
    FormsModule,
    RouterModule,
    HttpClientModule,
    NavbarModule,
    FooterModule,
    SidebarModule,
    AppRoutingModule
  ],
  declarations: [
    AppComponent,
    AdminLayoutComponent
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
