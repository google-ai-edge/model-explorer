import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  Inject,
  signal,
} from '@angular/core';
import {
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import {
  MAT_DIALOG_DATA,
  MatDialogModule,
  MatDialogRef,
} from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';

import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import {
  ConfigEditor,
  ConfigEditorType,
  ConfigValue,
  FileConfigEditor,
  NodeDataProviderExtension,
} from '../../common/types';

const UPLOAD_API_PATH = '/apipost/v1/upload';

declare interface UploadResponse {
  path: string;
}

/** Data for the RunTaskDialog. */
export interface RunNdpExtensionDialogData {
  extension: NodeDataProviderExtension;
}

export interface RunNdpExtensionDialogResult {
  runName: string;
  configValues: Record<string, ConfigValue>;
}

interface FileUploadStatus {
  fileName: string;
  status: FileUploadStatusType;
  error?: string;
}

enum FileUploadStatusType {
  READY,
  UPLOADING,
  UPLOADED,
  ERROR,
}

/** The drop down menu for add per-node data. */
@Component({
  standalone: true,
  selector: 'run-ndp-extension-dialog',
  imports: [
    CommonModule,
    MatDialogModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSlideToggleModule,
    MatTooltipModule,
    ReactiveFormsModule,
  ],
  templateUrl: 'run_ndp_extension_dialog.ng.html',
  styleUrls: ['./run_ndp_extension_dialog.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RunNdpExtensionDialog {
  readonly form = new FormGroup({});

  protected data: RunNdpExtensionDialogData;

  protected readonly ConfigEditorType = ConfigEditorType;
  protected readonly FileUploadStatusType = FileUploadStatusType;
  protected readonly configEditors: ConfigEditor[];
  // editor id -> status.
  protected readonly fileUploadStatus = signal<
    Record<string, FileUploadStatus>
  >({});
  protected readonly fileDragOver = signal<Record<string, boolean>>({});

  constructor(
    public dialogRef: MatDialogRef<RunNdpExtensionDialog>,
    @Inject(MAT_DIALOG_DATA) public dialogData: RunNdpExtensionDialogData
  ) {
    this.data = dialogData;

    // Add run name.
    this.configEditors = [
      {
        type: ConfigEditorType.TEXT_INPUT,
        id: 'runName',
        label: 'Run name',
        required: true,
        number: false,
        defaultValue: this.data.extension.id,
      },
      ...this.data.extension.configEditors,
    ];

    // Add config editors.
    for (const configEditor of this.configEditors) {
      const validators = [];
      if (configEditor.required) {
        validators.push(Validators.required);
      }
      switch (configEditor.type) {
        case ConfigEditorType.TEXT_INPUT:
          if (configEditor.number) {
            this.form.addControl(
              configEditor.id,
              new FormControl<number>(
                (configEditor.defaultValue as number) ?? 0,
                validators
              )
            );
          } else {
            this.form.addControl(
              configEditor.id,
              new FormControl<string>(
                (configEditor.defaultValue as string) ?? '',
                validators
              )
            );
          }
          break;
        case ConfigEditorType.TEXT_AREA:
        case ConfigEditorType.DROP_DOWN:
        case ConfigEditorType.FILE:
          this.form.addControl(
            configEditor.id,
            new FormControl<string>(
              (configEditor.defaultValue as string) ?? '',
              validators
            )
          );
          if (configEditor.type === ConfigEditorType.FILE) {
            this.initFileUploadStatus(configEditor.id);
          }
          break;
        case ConfigEditorType.COLOR_PICKER:
          this.form.addControl(
            configEditor.id,
            new FormControl<string>(
              (configEditor.defaultValue as string) ?? 'black',
              validators
            )
          );
          break;
        case ConfigEditorType.SLIDE_TOGGLE:
          this.form.addControl(
            configEditor.id,
            new FormControl<boolean>(
              (configEditor.defaultValue as boolean) ?? false,
              validators
            )
          );
          break;
        case ConfigEditorType.BUTTON_TOGGLE:
          this.form.addControl(
            configEditor.id,
            new FormControl<string[]>(
              (configEditor.defaultValue as string[]) ?? [],
              validators
            )
          );
          break;
        default:
          break;
      }
    }
  }

  async handleClickRun() {
    const runName = this.form.get('runName')?.value || this.data.extension.id;
    const result: RunNdpExtensionDialogResult = {
      runName,
      configValues: this.form.value,
    };
    // Convert the value of number-only text input from string to number.
    for (const editor of this.configEditors) {
      if (editor.type === ConfigEditorType.TEXT_INPUT && editor.number) {
        try {
          let numValue = Number(result.configValues[editor.id]);
          if (numValue != null && !isNaN(numValue)) {
            result.configValues[editor.id] = numValue;
          }
        } catch (e) {
          // Ignore.
        }
      }
    }
    this.dialogRef.close(result);
  }

  async handleUploadFile(editor: FileConfigEditor, input: HTMLInputElement) {
    const files = input.files;
    if (!files || files.length === 0) {
      return;
    }

    const file = files[0];
    await this.uploadFile(file, editor.id);
    input.value = '';
  }

  handleDragOver(event: DragEvent, editorId: string) {
    event.preventDefault();
    this.setDragOver(editorId, true);
  }

  handleDragLeave(event: DragEvent, editorId: string) {
    this.setDragOver(editorId, false);
  }

  handleDrop(event: DragEvent, editorId: string) {
    event.preventDefault();
    this.setDragOver(editorId, false);

    const files: File[] = [];
    if (event.dataTransfer?.items) {
      // Use DataTransferItemList interface to access the file(s)
      Array.from(event.dataTransfer.items).forEach((item, i) => {
        // If dropped items aren't files, reject them
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file) {
            files.push(file);
          }
        }
      });
    } else {
      // Use DataTransfer interface to access the file(s)
      files.push(...Array.from(event.dataTransfer?.files || []));
    }

    if (files.length > 0) {
      this.uploadFile(files[0], editorId);
    }
  }

  handleClickRemoveUpload(editorId: string) {
    this.setFormStringValue(editorId, '');
    this.fileUploadStatus.update((status) => {
      return {
        ...status,
        [editorId]: {
          fileName: '',
          status: FileUploadStatusType.READY,
        },
      };
    });
  }

  getFormStringValue(formName: string): string {
    return this.form.get(formName)?.value ?? '';
  }

  setFormStringValue(formName: string, value: string) {
    this.form.get(formName)?.setValue(value);
  }

  getFormStringArrayValue(formName: string): string[] {
    return this.form.get(formName)?.value ?? [];
  }

  getFormControl(formName: string): FormControl {
    return this.form.get(formName) as FormControl;
  }

  isFormRequired(formName: string): boolean {
    return this.form.get(formName)?.hasValidator(Validators.required) ?? false;
  }

  hasFormError(formName: string): boolean {
    const form = this.form.get(formName);
    if (!form) {
      return false;
    }
    return form.invalid && (form.dirty || form.touched);
  }

  toggleFormStringArrayValue(
    multiple: boolean,
    formName: string,
    value: string
  ) {
    if (!multiple) {
      this.form.get(formName)?.setValue([value]);
    } else {
      const curValue = this.getFormStringArrayValue(formName);
      if (curValue.includes(value)) {
        curValue.splice(curValue.indexOf(value), 1);
      } else {
        curValue.push(value);
      }
      this.form.get(formName)?.setValue(curValue);
    }
  }

  initFileUploadStatus(editorId: string) {
    this.fileUploadStatus.update((status) => {
      status[editorId] = {
        fileName: '',
        status: FileUploadStatusType.READY,
      };
      return { ...status };
    });
  }

  getFileUploadStatus(editorId: string): FileUploadStatusType {
    return this.fileUploadStatus()[editorId]?.status;
  }

  getFileUploadName(editorId: string): string {
    return this.fileUploadStatus()[editorId]?.fileName ?? '';
  }

  isDragOver(editorId: string): boolean {
    return this.fileDragOver()[editorId] ?? false;
  }

  get extension(): NodeDataProviderExtension {
    return this.data.extension;
  }

  get extensionName(): string {
    return this.data.extension.name;
  }

  private async uploadFile(file: File, editorId: string) {
    const fileName = file.name;
    this.setFileUploadingStatus(
      editorId,
      fileName,
      FileUploadStatusType.UPLOADING
    );
    const data = new FormData();
    data.append('file', file, file.name);
    const uploadResp = await fetch(UPLOAD_API_PATH, {
      method: 'POST',
      body: data,
    });
    if (!uploadResp.ok) {
      const error = await uploadResp.text();
      this.setFileUploadError(editorId, fileName, error);
    } else {
      const path = (JSON.parse(await uploadResp.text()) as UploadResponse).path;
      this.setFormStringValue(editorId, path);
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
    this.setFileUploadingStatus(
      editorId,
      fileName,
      FileUploadStatusType.UPLOADED
    );
  }

  private setFileUploadError(
    editorId: string,
    fileName: string,
    error: string
  ) {
    this.fileUploadStatus.update((status) => {
      if (status[editorId]) {
        status[editorId].error = error;
      } else {
        status[editorId] = {
          fileName,
          status: FileUploadStatusType.ERROR,
          error,
        };
      }
      return { ...status };
    });
  }

  private setFileUploadingStatus(
    editorId: string,
    fileName: string,
    status: FileUploadStatusType
  ) {
    this.fileUploadStatus.update((curStatus) => {
      return { ...curStatus, [editorId]: { status, fileName } };
    });
  }

  private setDragOver(editorId: string, dragOver: boolean) {
    this.fileDragOver.update((curStatus) => {
      return { ...curStatus, [editorId]: dragOver };
    });
  }
}
