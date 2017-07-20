function varargout = sensai(varargin)
% SENSAI M-file for sensai.fig
%      SENSAI, by itself, creates a new SENSAI or raises the existing
%      singleton*.
%
%      H = SENSAI returns the handle to a new SENSAI or the handle to
%      the existing singleton*.
%
%      SENSAI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SENSAI.M with the given input arguments.
%
%      SENSAI('Property','Value',...) creates a new SENSAI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before sensai_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to sensai_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help sensai

% Last Modified by GUIDE v2.5 04-Jul-2015 15:54:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @sensai_OpeningFcn, ...
                   'gui_OutputFcn',  @sensai_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before sensai is made visible.
function sensai_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to sensai (see VARARGIN)

% Choose default command line output for sensai
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes sensai wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = sensai_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     f1(x)                            %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f1x_Callback(hObject, eventdata, handles)
% hObject    handle to f1x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f1x as text
%        str2double(get(hObject,'String')) returns contents of f1x as a double
f1x = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(f1x))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function f1x_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f1x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     f2(x)                            %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f2x_Callback(hObject, eventdata, handles)
% hObject    handle to f2x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f2x as text
%        str2double(get(hObject,'String')) returns contents of f2x as a double
f2x = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(f2x))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function f2x_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f2x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     f3(x)                            %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f3x_Callback(hObject, eventdata, handles)
% hObject    handle to f3x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f3x as text
%        str2double(get(hObject,'String')) returns contents of f3x as a double
f3x = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(f3x))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function f3x_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f3x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     f4(x)                            %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f4x_Callback(hObject, eventdata, handles)
% hObject    handle to f4x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f4x as text
%        str2double(get(hObject,'String')) returns contents of f4x as a double
f4x = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(f4x))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function f4x_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f4x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%    Quantity of Interest              %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function QoI_Callback(hObject, eventdata, handles)
% hObject    handle to QoI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of QoI as text
%        str2double(get(hObject,'String')) returns contents of QoI as a double


% --- Executes during object creation, after setting all properties.
function QoI_CreateFcn(hObject, eventdata, handles)
% hObject    handle to QoI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%    User defined parameter            %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function cp_Callback(hObject, eventdata, handles)
% hObject    handle to cp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of cp as text
%        str2double(get(hObject,'String')) returns contents of cp as a double


% --- Executes during object creation, after setting all properties.
function cp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     String:  Create Matlab files using MuPAD %
%     Tag:     RunMuPAD                %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --- Executes on button press in RunMuPAD.
function RunMuPAD_Callback(hObject, eventdata, handles)
% hObject    handle to RunMuPAD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

warning off all;

rehash path

addpath('MuPadRoutines')
fprintf('RunMuPAD \n')
igui = get(handles.UseGUI,'Value');

if igui == 1;
    
    DIR = pwd;
    solution_only = get(handles.solution_only,'Value');
    xdim = str2num(get(handles.xdim,'String'));
    kdim = str2num(get(handles.kdim,'String'));
    fparam = [xdim,kdim];
    
    stype=3;
    if solutions_only == 1
        stype=1;
    end
    
    f1x=get(handles.f1x,'String');
    fvec(1,1:length(f1x)) = f1x;
    for i=2:xdim
        fi=strcat('f',num2str(i),'x');
        fix=eval( strcat( 'get(handles.',fi,', ''String'')' ) );
        fvec(i,1:length(fix))=fix;
    end
    
    qoi=get(handles.QoI,'String');
    cp=get(handles.cp,'String');
    
    NextGen = 0;
    R0_only = 0;
    imap = get(handles.map,'Value');
    % Input initial solution
    x1 = str2num(get(handles.x1,'String'));
    x2 = str2num(get(handles.x2,'String'));   
    x3 = str2num(get(handles.x3,'String'));
    x4 = str2num(get(handles.x4,'String'));
    x0=zeros(xdim,1);
    switch xdim
        case 1
            x0=[x1];
        case 2
            x0=[x1; x2];
        case 3
            x0=[x1; x2; x3];
        case 4
            x0=[x1; x2; x3; x4];
    end

    % Input parameters
    p1 = str2num(get(handles.p1,'String'));
    p2 = str2num(get(handles.p2,'String'));
    p3 = str2num(get(handles.p3,'String'));
    p4 = str2num(get(handles.p4,'String'));
    p5 = str2num(get(handles.p5,'String'));
    p6 = str2num(get(handles.p6,'String'));
    p0=zeros(kdim,1);
    switch kdim
        case 1
            param=[p1];
        case 2
            param=[p1; p2];
        case 3
            param=[p1; p2; p3];
        case 4
            param=[p1; p2; p3; p4];
       case 5
            param=[p1; p2; p3; p4; p5];
       case 6
            param=[p1; p2; p3; p4; p5; p6];
    end
    
else
    
    fprintf('RunMuPAD: implement user equations \n')
    userdirectory = get(handles.userfiles,'String');
    fprintf('User directory = %s \n', userdirectory)
    addpath(userdirectory)
    
    [fparam,fvec]=user_equations;         
    
    % Note: user_inputs is called only to get solution_only
    [DIR,JOB,imap,x0,param,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs;
    xdim=fparam(1);
    kdim=fparam(2);
    
    fprintf('RunMuPAD: implement user QoI \n')
    [qoi,qdim]=user_QoI;
    fparam(3)=qdim;
    
    fprintf('RunMuPAD: implement user parameters \n')
    [cp]=user_parameters;
    
    rmpath(userdirectory)
    
end

fprintf('RunMuPAD: xdim  = %3i \n', xdim)
fprintf('RunMuPAD: kdim  = %3i \n', kdim)
for i=1:xdim
    fprintf('RunMuPAD: f(%3i)  = %s \n', i,fvec(i,:))
end 
fprintf('RunMuPAD: cp  = %s \n', cp)

mm_interface(fparam,fvec,qoi,cp,qtype,stype,NextGen,x0,param,imap,R0_only)
pause(2);

fprintf('\n\n*********************************\n')
fprintf('Matlab files successfully created \n')
fprintf('*********************************\n\n')
msgbox('MATLAB files successfully created');

rmpath('MuPadRoutines')
guidata(hObject, handles);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     xdim                             %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function xdim_Callback(hObject, eventdata, handles)
% hObject    handle to xdim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of xdim as text
%        str2double(get(hObject,'String')) returns contents of xdim as a double
xdim = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(xdim))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function xdim_CreateFcn(hObject, eventdata, handles)
% hObject    handle to xdim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     variable #1                      %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x1_Callback(hObject, eventdata, handles)
% hObject    handle to x1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of x1 as text
%        str2double(get(hObject,'String')) returns contents of x1 as a double


% --- Executes during object creation, after setting all properties.
function x1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to x1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     variable #2                      %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x2_Callback(hObject, eventdata, handles)
% hObject    handle to x4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of x4 as text
%        str2double(get(hObject,'String')) returns contents of x4 as a double


% --- Executes during object creation, after setting all properties.
function x2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to x4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     variable #3                      %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x3_Callback(hObject, eventdata, handles)
% hObject    handle to x3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of x3 as text
%        str2double(get(hObject,'String')) returns contents of x3 as a double


% --- Executes during object creation, after setting all properties.
function x3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to x3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     variable #4                      %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x4_Callback(hObject, eventdata, handles)
% hObject    handle to x2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of x2 as text
%        str2double(get(hObject,'String')) returns contents of x2 as a double


% --- Executes during object creation, after setting all properties.
function x4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to x2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     kdim                             %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function kdim_Callback(hObject, eventdata, handles)
% hObject    handle to kdim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kdim as text
%        str2double(get(hObject,'String')) returns contents of kdim as a double
kdim = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(kdim))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function kdim_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kdim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     parameter #1                     %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p1_Callback(hObject, eventdata, handles)
% hObject    handle to p1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p1 as text
%        str2double(get(hObject,'String')) returns contents of p1 as a double
p1 = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(p1))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function p1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     parameter #2                     %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p2_Callback(hObject, eventdata, handles)
% hObject    handle to p2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p2 as text
%        str2double(get(hObject,'String')) returns contents of p2 as a double
p2 = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(p2))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function p2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     parameter #3                     %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p3_Callback(hObject, eventdata, handles)
% hObject    handle to p3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p3 as text
%        str2double(get(hObject,'String')) returns contents of p3 as a double
p3 = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(p3))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function p3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     parameter #4                     %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p4_Callback(hObject, eventdata, handles)
% hObject    handle to p4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p4 as text
%        str2double(get(hObject,'String')) returns contents of p4 as a double
p4 = get(hObject,'String')
%checks to see if input is empty. if so, default input1_editText to zero
if (isempty(p4))
     set(hObject,'String','0')
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function p4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     parameter #5                     %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p5_Callback(hObject, eventdata, handles)
% hObject    handle to p5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p5 as text
%        str2double(get(hObject,'String')) returns contents of p5 as a double

% --- Executes during object creation, after setting all properties.
function p5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     parameter #6                     %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p6_Callback(hObject, eventdata, handles)
% hObject    handle to p6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p6 as text
%        str2double(get(hObject,'String')) returns contents of p6 as a double

% --- Executes during object creation, after setting all properties.
function p6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     tfinal                           %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tfinal_Callback(hObject, eventdata, handles)
% hObject    handle to tfinal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tfinal as text
%        str2double(get(hObject,'String')) returns contents of tfinal as a double


% --- Executes during object creation, after setting all properties.
function tfinal_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tfinal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     String:  Execute Matlab file created by MuPAD  %
%     Tag:     RunMatlab               %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in RunMatlab.
function RunMatlab_Callback(hObject, eventdata, handles)
% hObject    handle to RunMatlab (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

rehash path

addpath('SolverRoutines', 'BifnRoutines', 'PlotRoutines')
fprintf('\n\n********* \n')
fprintf('RunMatlab \n')
fprintf('********* \n')

igui = get(handles.UseGUI,'Value');
imap = get(handles.map,'Value');

if igui==1
    
    DIR=pwd;
    JOB=get(handles.jobname,'String');
    
    xdim = str2num(get(handles.xdim,'String'));
    kdim = str2num(get(handles.kdim,'String'));
    fprintf('RunMatlab: xdim  = %3i \n', xdim)
    fprintf('RunMatlab: kdim = %3i \n', kdim)

    % Input initial solution
    x1 = str2num(get(handles.x1,'String'));
    x2 = str2num(get(handles.x2,'String'));   
    x3 = str2num(get(handles.x3,'String'));
    x4 = str2num(get(handles.x4,'String'));
    x0=zeros(xdim,1);
    switch xdim
        case 1
            x0=[x1];
        case 2
            x0=[x1; x2];
        case 3
            x0=[x1; x2; x3];
        case 4
            x0=[x1; x2; x3; x4];
    end
    fprintf('RunMatlab: x1  = %13.6e \n', x1)
    fprintf('RunMatlab: x2  = %13.6e \n', x2)
    fprintf('RunMatlab: x3  = %13.6e \n', x3)
    fprintf('RunMatlab: x4  = %13.6e \n', x4)

    % Input parameters
    p1 = str2num(get(handles.p1,'String'));
    p2 = str2num(get(handles.p2,'String'));
    p3 = str2num(get(handles.p3,'String'));
    p4 = str2num(get(handles.p4,'String'));
    p5 = str2num(get(handles.p5,'String'));
    p6 = str2num(get(handles.p6,'String'));
    p0=zeros(kdim,1);
    switch kdim
        case 1
            param=[p1];
        case 2
            param=[p1; p2];
        case 3
            param=[p1; p2; p3];
        case 4
            param=[p1; p2; p3; p4];
       case 5
            param=[p1; p2; p3; p4; p5];
       case 6
            param=[p1; p2; p3; p4; p5; p6];
    end
    fprintf('RunMatlab: p1  = %13.6e \n', p1)
    fprintf('RunMatlab: p2  = %13.6e \n', p2)
    fprintf('RunMatlab: p3  = %13.6e \n', p3)
    fprintf('RunMatlab: p4  = %13.6e \n', p4)   
    fprintf('RunMatlab: p5  = %13.6e \n', p5)
    fprintf('RunMatlab: p6  = %13.6e \n', p6)
    
    tfinal = str2num(get(handles.tfinal,'String'));

    % Hardwire certain options
    ilist=1:xdim;
    klist=1:kdim;
    
    % New section created 6-9-2010
    iplot_sensitivities = get(handles.plotsensitivities,'Value');
    iplot_elasticities = get(handles.plotelasticities,'Value');
    iplot_x_p_0 = get(handles.xwrtp,'Value');
    iplot_x_p = get(handles.xwrtps,'Value');
    iplot_x_ics = get(handles.xwrtics,'Value');
    iplot_qoi_p = get(handles.qoiwrtp,'Value');
    iplot_qoi_ics = get(handles.qoiwrtics,'Value');
    iplot_cp = get(handles.wrtcp,'Value');
    iplot_org_by_param = get(handles.xwrtpp,'Value');
    
    % Put the plot parameters into a structure
    iplot.sensitivities = iplot_sensitivities;
    iplot.elasticities = iplot_elasticities;
    iplot.dxdp.true = iplot_x_p_0;
    iplot.dxdp.param = iplot_org_by_param;
    iplot.dxdp.var = iplot_x_p;
    iplot.dxdz = iplot_x_ics;
    iplot.dqdp = iplot_qoi_p;
    iplot.dqdz = iplot_qoi_ics;
    iplot.cp = iplot_cp;
         
    % New option created 6-22-2010
    solution_only = get(handles.solution_only,'Value');
    
    NextGen=0;
    R0_only = 0;
    
end

if igui ~= 1

        userdirectory = get(handles.userfiles,'String');
        fprintf('Model defined in directory %s \n', userdirectory)
        addpath(userdirectory)
         
        [DIR,JOB,imap,x0,param,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs;
        [qoi,qdim]=user_QoI;
        [ilambda,imu,nu,ds,nstep]=user_bifndata;
        [eFIM,qFIM,Fdim,Fp,iFtimes,pest,pert]=user_FIMdata;
        [iplot,ilist,klist]=user_plotdata;

        rmpath(userdirectory)
  
end   

% Compute map/ode solution and its stability wrt parameters and initial conditions
gtype(DIR,JOB,imap,x0,param,tfinal,solntimes,ntsteps,qdim,...
       eFIM,qFIM,Fdim,Fp,iFtimes,pest,pert,...
       iplot,ilist,klist,...
       qtype,stype,NextGen,R0_only,...
       ilambda,imu,nu,ds,nstep);

rmpath('SolverRoutines', 'BifnRoutines', 'PlotRoutines')
guidata(hObject, handles);
clear all;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     Input from GUI button            %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in UseGUI.
function UseGUI_Callback(hObject, eventdata, handles)
% hObject    handle to UseGUI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of UseGUI

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     Map or ODE button                %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in map.
function map_Callback(hObject, eventdata, handles)
% hObject    handle to map (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of map

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     User Files                       %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function userfiles_Callback(hObject, eventdata, handles)
% hObject    handle to userfiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of userfiles as text
%        str2double(get(hObject,'String')) returns contents of userfiles as a double


% --- Executes during object creation, after setting all properties.
function userfiles_CreateFcn(hObject, eventdata, handles)
% hObject    handle to userfiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%     Job Name                         %
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function jobname_Callback(hObject, eventdata, handles)
% hObject    handle to jobname (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of jobname as text
%        str2double(get(hObject,'String')) returns contents of jobname as a double


% --- Executes during object creation, after setting all properties.
function jobname_CreateFcn(hObject, eventdata, handles)
% hObject    handle to jobname (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function Title_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Title (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in plotsensitivities.
function plotsensitivities_Callback(hObject, eventdata, handles)
% hObject    handle to plotsensitivities (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of plotsensitivities


% --- Executes on button press in plotelasticities.
function plotelasticities_Callback(hObject, eventdata, handles)
% hObject    handle to plotelasticities (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of plotelasticities


% --- Executes on button press in xwrtp.
function xwrtp_Callback(hObject, eventdata, handles)
% hObject    handle to xwrtp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of xwrtp


% --- Executes on button press in xwrtics.
function xwrtics_Callback(hObject, eventdata, handles)
% hObject    handle to xwrtics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of xwrtics


% --- Executes on button press in qoiwrtp.
function qoiwrtp_Callback(hObject, eventdata, handles)
% hObject    handle to qoiwrtp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of qoiwrtp


% --- Executes on button press in qoiwrtics.
function qoiwrtics_Callback(hObject, eventdata, handles)
% hObject    handle to qoiwrtics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of qoiwrtics


% --- Executes on button press in wrtcp.
function wrtcp_Callback(hObject, eventdata, handles)
% hObject    handle to wrtcp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of wrtcp


% --- Executes on button press in orgbyparam.
function orgbyparam_Callback(hObject, eventdata, handles)
% hObject    handle to orgbyparam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of orgbyparam


% --- Executes on button press in xwrtps.
function xwrtps_Callback(hObject, eventdata, handles)
% hObject    handle to xwrtps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of xwrtps


% --- Executes on button press in solution_only.
function solution_only_Callback(hObject, eventdata, handles)
% hObject    handle to solution_only (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of solution_only
