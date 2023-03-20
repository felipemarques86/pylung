import { TestBed } from '@angular/core/testing';

import { PylungService } from './pylung.service';

describe('PylungService', () => {
  let service: PylungService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PylungService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
