r0 := proc(NextGen,gparam,gtotal,x0,p0,imap)

local xname, sname;
  
begin

fd:=fopen("r0_matrix.m",Text,Write); 
fprint(Unquoted, fd, "function [R0,dR0dx,dR0dp,R0warnings] = r0_matrix(x,p) \n");
 
xdim:=gparam[1];
kdim:=gparam[2];

fd2 := fopen("MuPadRoutines/mm_interface.out",Append);
fprint(Unquoted, fd2, "\n R0.mu");

// Convert the strings into equations for manipulation
gs:=stringlib::split(gtotal,";");

for i from 1 to xdim do:
  gx[i]:= text2expr(gs[i]):
  fprint(Unquoted, fd2, "gx[",i,"] = ", gx[i]);   // for debugging
end_for:

// Expand the equations in order to isolate terms
for i from 1 to xdim do:
  gx[i] := expand(gx[i]):
  fprint(Unquoted, fd2, "gx[",i,"] = ",gx[i]);
end_for:
 
// Convert back into strings for manipulation
fprint(Unquoted, fd2, "Expanded string equations:");
for i from 1 to xdim do:
  gs[i] := expr2text(gx[i]);
  fprint(Unquoted, fd2, "gs[",i,"] = ", gs[i]);
end_for;
 
// Is NextGen a Matrix?  -- if NextGen is more than one state, then YES, but if just one state, then NO!
fprint(Unquoted, fd2, "Is NextGen a Matrix?  ", testtype(NextGen, Dom::Matrix));

// Identify all the non-infected classes as indicators of terms that belong in F
allx:={$ 1..xdim};   // all states 1 to xdim
InfSet:={};          // infected states

if testtype(NextGen, Dom::Matrix) = TRUE 
then 
  for i from 1 to linalg::matdim(NextGen)[1] do:
     InfSet := InfSet union {NextGen[i]};    
  end_for;
  NoInf := allx minus InfSet;       // non-infected states
  NoInf := coerce(NoInf,DOM_LIST);   // define it as a list
  fprint(Unquoted, fd2, "NoInf = ", NoInf(1));
else  // there is only one infected state!
  InfSet := {NextGen};
  NoInf := allx minus InfSet;
  NoInf := coerce(NoInf, DOM_LIST);
  fprint(Unquoted, fd2, "NoInf = ", NoInf(1));
end_if;
fprint(Unquoted, fd2, "How many elements in NoInf? ", nops(NoInf), ". NoInf[1]=", NoInf[1]);

singV := 0;    // this variable will check if V has any obvious singularities

if testtype(NextGen, Dom::Matrix) = TRUE then
  for e from 1 to linalg::matdim(NextGen)[1] do:
    if singV = 0 then   // proceed if there is are no singular terms
      calF[e] := 0;  calV[e] := 0;
      fprint(Unquoted, fd2, "calF[",e,"] = ", calF[e], "\ncalV[",e,"] = ", calV[e]);
      eqstring := gs[NextGen[e]];
      fprint(Unquoted, fd2, "eqstring[",e,"] = ", eqstring);
    
      if text2expr(eqstring) = 0 then    // the equation may be 0 -- ex: Pine12b g[7]
        // if the equation is 0, this will make V singular: do not try to compute V^{-1}
        singV:=NextGen[e];
      else
        fprint(Unquoted, fd2, "singV = ",singV);
    
        // Split up equation by + and - for the partial terms (eventually to determine if it belongs in F or V)
        splits := sort(_concat(stringlib::contains(eqstring, "+", IndexList),stringlib::contains(eqstring, "-",IndexList)));
        fprint(Unquoted, fd2, "splits = ",splits);
        fprint(Unquoted, fd2, "length of splits = ",nops(splits));
        terms[1] := substring(eqstring, 1..splits[1]-1);
        for i from 2 to nops(splits) do:
          terms[i] := substring(eqstring, splits[i-1] .. splits[i]-1);
        end_for;
        terms[nops(splits)+1] := substring(eqstring,splits[nops(splits)] .. length(eqstring));
        for i from 1 to nops(splits)+1 do:
          fprint(Unquoted, fd2, "terms[",i,"] = ",terms[i]);
        end_for;
      
        // If the first term starts with "-", then term[1] will be " " and should be removed
        if terms[1] = "" then
          for i from 1 to nops(terms)-1 do:
            terms[i]:=terms[i+1]:
          end_for;
          delete terms[nops(terms)];
          fprint(Unquoted, fd2, "Remove the first term (which is empty)");
          for i from 1 to nops(splits) do:
            fprint(Unquoted, fd2, "terms[",i,"] = ",terms[i]);
          end_for;
          fprint(Unquoted, fd2, "Number of terms = ",nops(terms));
        end_if;
      
        // Now we need to look through the indices of partial terms and see if there are any ('s to construct complete Terms
        bc := 1;     // counter for the size of the actual "big" terms once the () balance is accounted for
        for i from 1 to nops(terms) do:
          pflag := 0; nflag := 0;
          for j from 1 to length(terms[i]) do:  // loops over each element in term i to see if "(" appears
            if "(" = terms[i][j] then 
              pflag := pflag +1:                // flags the # of ( that appear in term i
            end_if;
            if ")" = terms[i][j] then
              nflag := nflag + 1:               // flags the number of ) that appear in term i
            end_if;
          end_for;
          nparens := pflag - nflag;             // number of parens sets missing a )
          fprint(Unquoted, fd2, "term ",i,", nparens = ",nparens);
        
          if nparens > 0 then
            k:= i+1;                // start searching for the balancing ) with the next (i+1) term
            while nparens > 0 do:
              for m from 1 to length(terms[k]) do:     // loops over remaining terms to balance out nparens
                if "(" = terms[k][m] then
                  nparens := nparens +1:
                end_if;
                if ")" = terms[k][m] then 
                  nparens := nparens -1;
                end_if;
              end_for;
              k:= k+1;
            end_while;
            ThisTerm := terms[i];
            for n from i+1 to k-1 do:                    // current term will include up to the balancing term
              ThisTerm := _concat(ThisTerm, terms[n]):
            end_for;
            fprint(Unquoted, fd2, "ThisTerm = ", ThisTerm);
          
            Bterms[bc] := ThisTerm:          // only problem left is + and - at beginnings
            i := k-1;
          else
            ThisTerm := terms[i];
            Bterms[bc] := ThisTerm:          
          end_if;
          bc := bc +1;    // update counter for "big" terms
        end_for;
        for i from 1 to nops(Bterms) do:
          fprint(Unquoted, fd2, "Bterms[",i,"] = ", Bterms[i]);
        end_for;
        
        // Don't need to check if there is only one Bterm  (new syntax makes it nicer!)
        
        // Put the whole Bterms back into equation form for manipulation -- Eterms
        for i from 1 to nops(Bterms) do:
          Eterms[i]:=text2expr(Bterms[i]):
          fprint(Unquoted, fd2, "Eterms[",i,"] = ", Eterms[i]);
        end_for;
        
        // In case the population is scaled by the total population, hide disease-free states that are just scaling factors
        // This is necessary as the occurrence of a disease-free state is the determining factor if that term belongs in F or V
        // Define N = sum of all states
        var := 0:
        for i from 1 to xdim do:
          var := var + x[i]:
        end_for:
        fprint(Unquoted, fd2, "N = ", var);
        
        // Substitute N into all equations and expand the equations in order to isolate terms
        for i from 1 to nops(Eterms) do:
          EtermsN[i] := subs(Eterms[i], var = N):
          BtermsN[i] := expr2text(EtermsN[i]):
          fprint(Unquoted, fd2, "BtermsN[",i,"] = ",BtermsN[i]);
        end_for:
         
        // See if the Terms belong in F or V -- search through NoInf
        Vterms := matrix(1, nops(Eterms)):
        Fterms := matrix(1, nops(Eterms)):
  
      
        for s from 1 to nops(NoInf) do:
          SPattern[s] := _concat("x[", NoInf[s], "]"):
        end_for: 
        
        for i from 1 to nops(SPattern) do:
          fprint(Unquoted, fd2, "SPattern[",i,"] = ",SPattern[i]);
        end_for;
         
        for i from 1 to nops(Eterms) do:
          S := 0; 
          for s from 1 to nops(NoInf) do:
            SIndex := stringlib::contains(BtermsN[i], SPattern[s], IndexList);  // search through BtermsN to decide is Eterms belongs in F or V
            if nops(SIndex) = 0 then  // don't add to S  
            else S := 1; 
            end_if;
          end_for:
          fprint(Unquoted, fd2, "Is an SPattern in term ",i,"?  ", S);
          
           // another criterion for belonging in calF is if the entire term is parameter(s)
           // search through to see if there is any "x" in this term
          XIndex := stringlib::contains(BtermsN[i], "x", IndexList); 
          if nops(XIndex) = 0 then S := 1  
          end_if;
          fprint(Unquoted, fd2, "Is term ",i," only parameters (OR SPattern in it)?  ",S);
          
          if S = 0 then  // this term belongs in V and should be negated if ODE
            if imap = 0 then
              Vterms[i] := -(Eterms[i]);
              fprint(Unquoted, fd2, "Vterms[",i,"] = ", Vterms[i]);
            else
              Vterms[i] := Eterms[i];
              fprint(Unquoted, fd2, "Vterms[",i,"] = ", Vterms[i]);
            end_if;           
          else 
            Fterms[i] := Eterms[i]:
            fprint(Unquoted, fd2, "Fterms[",i,"] = ", Fterms[i]);
          end_if;  
        end_for:
        
        // Combine FTerms and VTerms into mathcal(F) and mathcal(V)
        calF[e] := 0: calV[e] := 0:
        for i from 1 to nops(Bterms) do:
          calF[e] := calF[e]+Fterms[i]:
          calV[e] := calV[e]+Vterms[i]:
          fprint(Unquoted, fd2, "\ncalF[",e,"] = ", calF[e]);
          fprint(Unquoted, fd2, "calV[",e,"] = ", calV[e]);
        end_for;
        
      end_if;
    end_if;
    delete terms, Bterms, Eterms, BtermsN, Fterms, Vterms;
  end_for;
  
  
else  
  // only one infected equation-- F and V will be 1x1 matrices
  e:=1;
  if singV = 0 then   // proceed if there is are no singular terms
    calF[e] := 0;  calV[e] := 0;
    fprint(Unquoted, fd2, "calF = ", calF[e], "\ncalV = ", calV[e]);
    eqstring := gs[NextGen];
    fprint(Unquoted, fd2, "eqstring = ", eqstring);
    
    if text2expr(eqstring) = 0 then    // the equation may be 0 -- ex: Pine12b g[7]
      // if the equation is 0, this will make V singular: do not try to compute V^{-1}
      singV:=NextGen[e];
    else
      fprint(Unquoted, fd2, "singV = ",singV);
    
      // Split up equation by + and - for the partial terms (eventually to determine if it belongs in F or V)
      splits := sort(_concat(stringlib::contains(eqstring, "+", IndexList),stringlib::contains(eqstring, "-",IndexList)));
      fprint(Unquoted, fd2, "splits = ",splits);
      fprint(Unquoted, fd2, "length of splits = ",nops(splits));
      terms[1] := substring(eqstring, 1..splits[1]-1);
      for i from 2 to nops(splits) do:
        terms[i] := substring(eqstring, splits[i-1] .. splits[i]-1);
      end_for;
      terms[nops(splits)+1] := substring(eqstring,splits[nops(splits)] .. length(eqstring));
      for i from 1 to nops(splits)+1 do:
        fprint(Unquoted, fd2, "terms[",i,"] = ",terms[i]);
      end_for;
    
      // If the first term starts with "-", then term[1] will be " " and should be removed
      if terms[1] = "" then
        for i from 1 to nops(terms)-1 do:
          terms[i]:=terms[i+1]:
        end_for;
        delete terms[nops(terms)];
        fprint(Unquoted, fd2, "Remove the first term (which is empty)");
        for i from 1 to nops(splits) do:
          fprint(Unquoted, fd2, "terms[",i,"] = ",terms[i]);
        end_for;
        fprint(Unquoted, fd2, "Number of terms = ",nops(terms));
      end_if;
    
      // Now we need to look through the indices of partial terms and see if there are any ('s to construct complete Terms
      bc := 1;     // counter for the size of the actual "big" terms once the () balance is accounted for
      for i from 1 to nops(terms) do:
        pflag := 0; nflag := 0;
        for j from 1 to length(terms[i]) do:  // loops over each element in term i to see if "(" appears
          if "(" = terms[i][j] then 
            pflag := pflag +1:                // flags the # of ( that appear in term i
          end_if;
          if ")" = terms[i][j] then
            nflag := nflag + 1:               // flags the number of ) that appear in term i
          end_if;
        end_for;
        nparens := pflag - nflag;             // number of parens sets missing a )
        fprint(Unquoted, fd2, "term ",i,", nparens = ",nparens);
      
        if nparens > 0 then
          k:= i+1;                // start searching for the balancing ) with the next (i+1) term
          while nparens > 0 do:
            for m from 1 to length(terms[k]) do:     // loops over remaining terms to balance out nparens
              if "(" = terms[k][m] then
                nparens := nparens +1:
              end_if;
              if ")" = terms[k][m] then 
                nparens := nparens -1;
              end_if;
            end_for;
            k:= k+1;
          end_while;
          ThisTerm := terms[i];
          for n from i+1 to k-1 do:                    // current term will include up to the balancing term
            ThisTerm := _concat(ThisTerm, terms[n]):
          end_for;
          fprint(Unquoted, fd2, "ThisTerm = ", ThisTerm);
        
          Bterms[bc] := ThisTerm:          // only problem left is + and - at beginnings
          i := k-1;
        else
          ThisTerm := terms[i];
          Bterms[bc] := ThisTerm:          
        end_if;
        bc := bc +1;    // update counter for "big" terms
      end_for;
      for i from 1 to nops(Bterms) do:
        fprint(Unquoted, fd2, "Bterms[",i,"] = ", Bterms[i]);
      end_for;
      
      // Don't need to check if there is only one Bterm  (new syntax makes it nicer!)
      
      // Put the whole Bterms back into equation form for manipulation -- Eterms
      for i from 1 to nops(Bterms) do:
        Eterms[i]:=text2expr(Bterms[i]):
        fprint(Unquoted, fd2, "Eterms[",i,"] = ", Eterms[i]);
      end_for;
      
      // In case the population is scaled by the total population, hide disease-free states that are just scaling factors
      // This is necessary as the occurrence of a disease-free state is the determining factor if that term belongs in F or V
      // Define N = sum of all states
      var := 0:
      for i from 1 to xdim do:
        var := var + x[i]:
      end_for:
      fprint(Unquoted, fd2, "N = ", var);
      
      // Substitute N into all equations and expand the equations in order to isolate terms
      for i from 1 to nops(Eterms) do:
        EtermsN[i] := subs(Eterms[i], var = N):
        BtermsN[i] := expr2text(EtermsN[i]):
        fprint(Unquoted, fd2, "BtermsN[",i,"] = ",BtermsN[i]);
      end_for:
       
      // See if the Terms belong in F or V -- search through NoInf
      Vterms := matrix(1, nops(Eterms)):
      Fterms := matrix(1, nops(Eterms)):

      for s from 1 to nops(NoInf) do:
        SPattern[s] := _concat("x[", NoInf[s], "]"):
      end_for:
      
      for i from 1 to nops(SPattern) do:
        fprint(Unquoted, fd2, "SPattern[",i,"] = ",SPattern[i]);
      end_for;
      
      for i from 1 to nops(Eterms) do:
        S := 0; 
        for s from 1 to nops(NoInf) do:
          SIndex := stringlib::contains(BtermsN[i], SPattern[s], IndexList);  // search through BtermsN to decide is Eterms belongs in F or V
          if nops(SIndex) = 0 then  // don't add to S  
          else S := 1; 
          end_if;
        end_for:
        fprint(Unquoted, fd2, "Is an SPattern in term ",i,"?  ", S);
        
         // another criterion for belonging in calF is if the entire term is parameter(s)
         // search through to see if there is any "x" in this term
        XIndex := stringlib::contains(BtermsN[i], "x", IndexList); 
        if nops(XIndex) = 0 then S := 1  
        end_if;
        fprint(Unquoted, fd2, "Is term ",i," only parameters (OR SPattern in it)?  ",S);
        
        if S = 0 then  // this term belongs in V and should be negated if ODE
          if imap = 0 then
            Vterms[i] := -(Eterms[i]);
            fprint(Unquoted, fd2, "Vterms[",i,"] = ", Vterms[i]);
          else
            Vterms[i] := Eterms[i];
            fprint(Unquoted, fd2, "Vterms[",i,"] = ", Vterms[i]);
          end_if;           
        else 
          Fterms[i] := Eterms[i]:
          fprint(Unquoted, fd2, "Fterms[",i,"] = ", Fterms[i]);
        end_if;  
      end_for:
      
      // Combine FTerms and VTerms into mathcal(F) and mathcal(V)
      calF[e] := 0: calV[e] := 0:
      for i from 1 to nops(Bterms) do:
        calF[e] := calF[e]+Fterms[i]:
        calV[e] := calV[e]+Vterms[i]:
        fprint(Unquoted, fd2, "\ncalF[",e,"] = ", calF[e]);
        fprint(Unquoted, fd2, "calV[",e,"] = ", calV[e]);
      end_for;
      
    end_if;
  end_if;
end_if;

    
// Compute the Jacobians F and V evaluated at disease-free equilibrium
if testtype(NextGen, Dom::Matrix) = TRUE then
  FEqs := expr2text(calF[1]): 
  VEqs := expr2text(calV[1]): 
  Vars := _concat("x[",NextGen[1],"]"):  
  for i from 2 to linalg::matdim(NextGen)[1] do:
    FEqs := _concat(FEqs,",",expr2text(calF[i])): 
    VEqs := _concat(VEqs,",",expr2text(calV[i])): 
    Vars := _concat(Vars, ", x[", NextGen[i], "]"):
  end_for:
else
  FEqs := expr2text(calF[1]):
  VEqs := expr2text(calV[1]):
  Vars := _concat("x[",NextGen,"]");
end_if;

fprint(Unquoted, fd2, "\nFEqs = ", FEqs);
fprint(Unquoted, fd2, "VEqs = ", VEqs);
fprint(Unquoted, fd2, "Vars = ", Vars);

F := linalg::jacobian([text2expr(FEqs)], [text2expr(Vars)] ): 
V := linalg::jacobian([text2expr(VEqs)], [text2expr(Vars)] ):
fprint(Unquoted, fd2, "\nF = ",F);
fprint(Unquoted, fd2, "V = ",V,"\n");

// Force infection classes to be 0 for the user for evaluation of eigenvalues
xeq:=x0:
if testtype(NextGen, Dom::Matrix) = FALSE then
   xeq[NextGen]:=0:
else
  for i from 1 to linalg::matdim(NextGen)[1] do:
    xeq[NextGen[i]]:=0:      
  end_for:
end_if;
fprint(Unquoted, fd2, "x0 = ", x0);
fprint(Unquoted, fd2, "xeq = ", xeq);
 
for i from 1 to xdim do:
  fprint(Unquoted, fd2, "gx[",i,"] = ", gx[i]);
end_for:

// Test for singular V beyond "zero" classes (ex: incorrect placement of terms in F when they belong in V)
if V^(-1) = FAIL and singV = 0 then
  singV := 100;    // singV identifies the equation that is identically 0.  If singV = 100, a different warning will display.
end_if;

if singV = 0 then
  // Compute R0 symbolically to take derivatives -- different for maps and odes
  if imap = 1 then     
    if testtype(NextGen, Dom::Matrix) = TRUE then
      NG := F*(matrix::identity(linalg::matdim(NextGen)[1]) -V)^(-1):
    else
      NG := F*(matrix::identity(nops(NextGen))-V)^(-1):
    end_if;
  else
    NG := F*(V^(-1));
  end_if;
  fprint(Unquoted, fd2, "Next Generation Matrix: ", NG);
  
  Evals := linalg::eigenvalues(NG);
  evals := evalAt(Evals, {p = p0, x = xeq});
  fprint(Unquoted, fd2, "Numerical eigenvalues: ", evals);

  // MuPad is not as efficient as Maple.  If the NG matrix is too complicated, the eigenvalues
  // will be given by a "RootOf" expression and can not be evaluated.  If this occurs, compute
  // the numerical R0 via the numerical F and V matrices.  Sensitivities will not be available.
  
  // If R0 is a RootOf expression, find the degree of the polynomial -- kept in case the derivatives of RootOf() [themselves not in RootOf() form] can be obtained in a later version
  str:= expr2text(evals);     // the eigenvalues
  fprint(Unquoted, fd2, "Number of elements in the string of eigenvalues: ",length(str));
  if length(str) > 5 then
    rstr := substring(str,1..6);  // "RootOf" some polynomial?
  else
    rstr := str:
  end_if;
  if rstr = "RootOf" then
    pstr := substring(str, 8..stringlib::contains(str,",",Index)-1);  // the polynomial as a string
    pol := text2expr(pstr);  // the polynomial as an expression
    N := degree(pol);        // N = number of eigenvalues
    warning0 := 1:           // warning that R0 is a RootOf() expression
  else
    N := nops(evals);         // N = number of eigenvalues
    warning0 := 0:           // R0 is not a RootOf() expression- no warning
  end_if;
  fprint(Unquoted, fd2, "N = ", N);
  fprint(Unquoted, fd2, "RootOf? = ", rstr);
  fprint(Unquoted, fd, "warning0 = ", warning0,";\n");
  
  if warning0 = 1 then    // do everything numerically!
    F:= evalAt(F, {p=p0, x=xeq});
    V:= evalAt(V, {p=p0, x=xeq});
    if imap = 1 then     
      if testtype(NextGen, Dom::Matrix) = TRUE then
        NG := F*(matrix::identity(linalg::matdim(NextGen)[1]) -V)^(-1):
      else
        NG := F*(matrix::identity(nops(NextGen))-V)^(-1):
      end_if;
    else
      NG := F*(V^(-1));
    end_if;
    fprint(Unquoted, fd2, "Numerical Next Generation Matrix: ", NG);
    
    Evals := linalg::eigenvalues(NG);
    evals := simplify(evalAt(Evals, {p = p0, x = xeq}));
    fprint(Unquoted, fd2, "Numerical eigenvalues: ", evals);
  end_if;
  
  // Finding R0 symbolically requires the values of r0 numerically as a maximum eigenvalue
  ind := 0: r0 := 0: 
  for i from 1 to nops(evals) do:      // replace nops(evals) with N in later version if diff(RootOf()) is acceptable
    if abs(float(evals[i])) > r0 then 
      r0 := abs(float(evals[i])): 
      ind := i:
    end_if;
  end_for:
  if ind = 0 then    // this will occur if all eigenvalues are negligible
    ind:=1;          
  end_if;
  R0 := Evals[ind]:
  r4 := 0:     // no warning for singular matrix V
else
  R0 := 0:     // so that it outputs something
  r0 := 0;
  fprint(Unquoted, fd, "warning0 = 0;\n"):
end_if;
fprint(Unquoted, fd2, "R0 = ", R0);
fprint(Unquoted, fd2, "r0 = ", r0);


// Write R0 to the MATLAB file r0_matrix.m
xname := text2expr("R0");
fprint(Unquoted, fd, generate::MATLAB(xname = R0));


// Include sensitivity equations for iteration in MATLAB
for i from 1 to xdim do:
   dR0dx[i] := diff(R0,x[i]):
end_for:

for j from 1 to kdim do:
   dR0dp[j] := diff(R0,p[j]):
end_for:

for i from 1 to xdim do:
   sname := _concat("dR0dx",i);
   xname := text2expr(sname);
   fprint(NoNL, fd, generate::MATLAB(xname = dR0dx[i]));
end_for:

for i from 1 to kdim do:
    sname := _concat("dR0dp",i);
    xname := text2expr(sname);
    fprint(NoNL, fd, generate::MATLAB(xname = dR0dp[i]));
end_for:

fprint(Unquoted, fd, "");

// R0=R0(x,p) so dR0dx and dR0dp should be iterated!
for i from 1 to xdim do:
    fprint(Unquoted, fd, "dR0dx(",i,") = dR0dx",i,";");
end_for:

for k from 1 to kdim do:
    sname := cat("dR0dp",sk);
    fprint(Unquoted, fd, "dR0dp(",k,") = dR0dp",k,";");
end_for:

fprint(Unquoted, fd, "");




//////////////////////////////////////////////////////////////////////////////
// Does the theorem hold?  Test the following to see if this is a valid R0: //
//////////////////////////////////////////////////////////////////////////////

// Requirement 1:  the spectral radius of V must be less than 1 (MAPS only)

Ftest := evalAt(F, {p = p0, x = x0}):
Vtest := evalAt(V, {p = p0, x = x0}):
r1vals := linalg::eigenvalues(Vtest): 
r1spec := max(abs(r1vals)):
fprint(Unquoted, fd2, "Spetral radius of Transition matrix (needs to be < 1) = ",r1spec);

r1 := 0: 
if r1spec > 1 then 
   r1 := 1:
end_if;

if imap = 0 then
   r1:=0:        // It doesn't matter if p(T) < 1 for ODE models
end_if;

sname := "warning1";
xname := text2expr(sname);
fprint(NoNL, fd, generate::MATLAB(xname = r1));



// Requirement 2:  F >= 0 and V>=0 (MAPS only.  V may be negative for ODE models)

r2F:=0:  r2V:=0:

for i from 1 to linalg::matdim(Ftest)[1] do:
   for j from 1 to linalg::matdim(Ftest)[2] do:
      if Ftest[i,j] < 0 then 
         r2F := 1:
      end_if;
      if Vtest[i,j] < 0 then 
         r2V := 1:
      end_if;
   end_for;
end_for;

if imap = 0 then
   r2V:=0:        // It doesn't matter if V is negative for ODE models
end_if;

xname := text2expr("warning2_F");
fprint(NoNL, fd, generate::MATLAB(xname = r2F));
xname := text2expr("warning2_V");
fprint(NoNL, fd, generate::MATLAB(xname = r2V));




// Requirement 3: The DFE must be stable in the absence of disease.  
// MAPS: For the non-"NextGen" states, the spectral radius of the Jacobian must be less than 1.  
// ODES: (A5) The entire Jacobian must have all negative real part eigenvalues if NextGen states = 0.

if imap = 1 then
    
  Jvars := _concat("x[", NoInf[1], "]"):
  Jeqs := expr2text(gx[NoInf[1]]):

  for i from 2 to nops(NoInf) do:
    Jvars := _concat(Jvars, ", x[", NoInf[i], "]"):
    Jeqs := _concat(Jeqs, ",",expr2text(gx[NoInf[i]])):
  end_for:
  
  fprint(Unquoted, fd2, "JEqs = ", Jeqs);
  fprint(Unquoted, fd2, "Jvars = ", Jvars);

  Cmat := linalg::jacobian([text2expr(Jeqs)], [text2expr(Jvars)]):
  Cval := evalAt(Cmat,{p=p0,x=xeq}):            // we don't need algebraic eigenvalues, just numerical!
  r3vals := linalg::eigenvalues(Cval):
  r3spec := max(abs(r3vals)):
  
  for i from 1 to linalg::matdim(Cmat)[1] do:
    for j from 1 to linalg::matdim(Cmat)[2] do:
      fprint(Unquoted, fd2, "Cval[",i,",",j,"] = ", Cval[i,j]);
    end_for;  
  end_for;
  fprint(Unquoted, fd2, "rho(C) = ", r3spec);

  r3 := 0: 
  if r3spec > 1 then 
    r3 := 1:
  end_if;
 
else
  
  if testtype(NextGen, Dom::Matrix) = FALSE then
    gx[NextGen]:=gx[NextGen]-calF[1];       // Set cal{F} = 0 by getting rid of it
  else
    for i from 1 to linalg::matdim(NextGen)[1] do:   
      gx[NextGen[i]]:=gx[NextGen[i]]-calF[i]:    // Set cal{F} = 0 by getting rid of it
    end_for;
  end_if;
  fprint(Unquoted, fd2, "gx without cal{F}: (this will only affect the states in NextGen)");
  for i from 1 to xdim do:
    fprint(Unquoted, fd2, "gx[",i,"] = ", gx[i]);
  end_for:
 
  Mvars := "x[1]";
  Meqs := expr2text(gx[1]);
  
  for i from 2 to xdim do:
    Mvars := _concat(Mvars, ", x[", i, "]"):
    Meqs := _concat(Meqs,",",expr2text(gx[i])):
  end_for:
  
  fprint(Unquoted, fd2, "Mvars = ", Mvars); 
  fprint(Unquoted, fd2, "Meqs = ", Meqs);

  Mmat := linalg::jacobian([text2expr(Meqs)], [text2expr(Mvars)] ):
  Mval := evalAt(Mmat,{p=p0,x=xeq}):          // we don't need algebraic eigenvalues, just numerical!
  r3vals := linalg::eigenvalues(Mval):
  r3vals := Re(r3vals):
  
  for i from 1 to linalg::matdim(Mmat)[1] do:
    for j from 1 to linalg::matdim(Mmat)[2] do:
     fprint(Unquoted, fd2, "Mval[",i,",",j,"] = ", Mval[i,j]);
    end_for;  
  end_for;
  fprint(Unquoted, fd2, "rho(entire system M) = ", r3vals);
  
  r3:=0:
  for i from 1 to nops(r3vals) do:
    if r3vals[i]>0 then
      r3:=1:
    end_if;
  end_for:
    
end_if;

xname := text2expr("warning3");
fprint(NoNL, fd, generate::MATLAB(xname = r3));
xname := text2expr("warning4");
fprint(NoNL, fd, generate::MATLAB(xname = singV));



// Requirement 4: (A4) for ODEs: if x is disease-free, it remains disease-free: that is, calF = 0

r5 := 0:
if imap = 0 then
  // use a vector with zero disease-states and 1 non-disease states
  //y := [0*i $ i=1..xdim]:
  y:= matrix(1,xdim);
  for i from 1 to nops(NoInf) do:
    y[NoInf[i]] := 1:
  end_for:
  fprint(Unquoted, fd2, "y (initial conditions with non-disease forced in)");
  for i from 1 to xdim do:
    fprint(Unquoted, fd2, "y[",i,"] = ",y[i]); 
  end_for;
  
  if testtype(NextGen, Dom::Matrix) = FALSE then
    fprint(Unquoted, fd2, "calF with disease introduced = ", calF[1], " = ", evalAt(calF[1], {p=p0, x=y}));
    if evalAt(calF[1],{p=p0,x=y})>0 then
      r5:= NextGen:
    end_if;
  else
    for i from 1 to linalg::matdim(NextGen)[1] do:
      fprint(Unquoted, fd2, "calF[",i,"] with disease introduced = ", calF[i], " = ", evalAt(calF[i], {p = p0, x = y}));
      if evalAt(calF[i], {p = p0, x = y}) > 0 then
        r5 := NextGen[i]:
      end_if;
    end_for:
  end_if;
end_if;

xname := text2expr("warning5");
fprint(Unquoted, fd, generate::MATLAB(xname = r5)); 

// Put warning statements into a MATLAB structure
fprint(Unquoted, fd, "\nR0warnings.w0 = warning0;");
fprint(Unquoted, fd, "R0warnings.w1 = warning1;");
fprint(Unquoted, fd, "R0warnings.w2F = warning2_F;");
fprint(Unquoted, fd, "R0warnings.w2V = warning2_V;");
fprint(Unquoted, fd, "R0warnings.w3 = warning3;");
fprint(Unquoted, fd, "R0warnings.w4 = warning4;");
fprint(Unquoted, fd, "R0warnings.w5 = warning5;");

fprint(Unquoted, fd, "");
fprint(Unquoted, fd2, "\n");
fprint(NoNL, fd, "end");

fclose(fd);
fclose(fd2);

end_proc;

