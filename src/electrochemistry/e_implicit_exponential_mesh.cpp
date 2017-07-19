
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/tuple.hpp>
#include <math.h> 
#include "e_implicit_exponential_mesh.hpp"
#include <iostream>
#include <exception>

namespace pints {

struct BCfun {
        BCfun(const double h0,
                const double Cdl,
                const double f1,
                const double e1,
                const double Eapp1,
                const double Eapp0,
                const double Ru,
                const double alpha,
                const double E0,
                const double dt,
                const double Itot0,
                const double k0):
            h0(h0),Cdl(Cdl),f1(f1),e1(e1),Eapp1(Eapp1),Eapp0(Eapp0),
            Ru(Ru),alpha(alpha),E0(E0),dt(dt),Itot0(Itot0),k0(k0)
        {}

        boost::math::tuple<double,double> operator()(const double Itot1) const {
            return boost::math::make_tuple(residual(Itot1),residual_gradient(Itot1));
        }
        //double operator()(const double Itot1) const {
        //    return residual(Itot1);
        //}
        double residual(const double Itot1) const {
            const double tmpIf = If(Itot1);
            const double exptmp = Eapp1 - Itot1*Ru - E0;
            const double U0 = (f1 - h0*tmpIf)/(1-e1);
            return tmpIf - k0*(U0*std::exp((1-alpha)*exptmp) - (1-U0)*std::exp(-alpha*exptmp));
        }

        double If(const double Itot1) const {
            const double deltaEeff = (Eapp1-Itot1*Ru) - (Eapp0-Itot0*Ru);
            const double Ic = Cdl*deltaEeff/dt;
            return Itot1 - Ic;
        }

        double If2(const double Itot1) const {
            const double tmpIf = If(Itot1);
            const double exptmp = Eapp1 - Itot1*Ru - E0;
            const double U0 = (f1 - h0*tmpIf)/(1-e1);
            return k0*(U0*std::exp((1-alpha)*exptmp) - (1-U0)*std::exp(-alpha*exptmp));
        }

        double residual_gradient(const double Itot1) const {
            const double tmpIf = If(Itot1);
            const double exptmp = Eapp1 - Itot1*Ru - E0;
            const double U0 = (f1 - h0*tmpIf)/(1-e1);
            const double tmp = Cdl*Ru/dt;
            const double dU0dItot1 = -h0*(1+tmp)/(1-e1);
            const double exp1 = std::exp((1-alpha)*exptmp);
            const double exp2 = std::exp(-alpha*exptmp);
            return 1 + tmp - k0*(dU0dItot1*(exp1+exp2) + U0*Ru*(alpha-1)*exp1 - (1-U0)*alpha*Ru*exp2);
        }

        const double h0,Cdl,f1,e1,Eapp1,Eapp0,Ru,alpha,E0,dt,Itot0,k0;
    };

struct TolFun {
    TolFun(const double tol):tol(tol) {}
    double operator()(const double min, const double max) const {
        return (max-min) < tol;
    } 
    const double tol;
};

void e_implicit_exponential_mesh(map& params, vector& Itot, vector& t) {
    const double k0 = get(params,std::string("k0"),35.0);
    const double alpha = get(params,std::string("alpha"),0.5);
    const double Cdl = get(params,std::string("Cdl"),0.0037);
    const double Ru = get(params,std::string("Ru"),2.74);
    const double E0 = get(params,std::string("E0"),0.0);
    const double dE = get(params,std::string("dE"),0.1);
    const int Nx = get(params,std::string("Nx"),300.0);
    const int Nt = get(params,std::string("Nt"),200.0);
    const int startn = get(params,std::string("startn"),0.0);
    const double Estart = get(params,std::string("Estart"),-10.0);
    const double Ereverse = get(params,std::string("Ereverse"),10.0);
    if (Ereverse < Estart) throw std::runtime_error("Ereverse must be greater than Estart");
    const double pi = boost::math::constants::pi<double>();
    const double omega = get(params,std::string("omega"),2*pi);
    const double phase = get(params,std::string("phase"),0.0);

#ifndef NDEBUG
    std::cout << "Running e_implicit_exponential_mesh with parameters:"<<std::endl;
    std::cout << "\tk0 = "<<k0<<std::endl;
    std::cout << "\talpha = "<<alpha<<std::endl;
    std::cout << "\tCdl = "<<Cdl<<std::endl;
    std::cout << "\tRu = "<<Ru<<std::endl;
    std::cout << "\tE0 = "<<E0<<std::endl;
    std::cout << "\tdE = "<<dE<<std::endl;
    std::cout << "\tNx = "<<Nx<<std::endl;
    std::cout << "\tNt = "<<Nt<<std::endl;
    std::cout << "\tstartn = "<<startn<<std::endl;
    std::cout << "\tEstart = "<<Estart<<std::endl;
    std::cout << "\tEreverse = "<<Ereverse<<std::endl;
    std::cout << "\tomega = "<<omega<<std::endl;
    std::cout << "\tphase = "<<phase<<std::endl;
    std::cout << "\tdE= "<<dE<<std::endl;
#endif


    const TolFun tol(1e-20);
    const int digits_accuracy = std::numeric_limits<double>::digits*2/3;
    const double max_iterations = 100;

    //set up spatial mesh
    const double Xmax = 20;
    const double r = 1.04;
    const double h0 = Xmax*(1-r)/(1-pow(r,Nx+1));
#ifndef NDEBUG
    std::cout << "\th0= "<<h0<<std::endl;
#endif
    vector h(Nx+1);
    double sum;
    for (int i=0; i<=Nx; i++) {
        h[i] = pow(r,i)*h0;
        sum += h[i];
    }

    //set up temporal mesh
    const double dt = (1.0/Nt)*2*pi/omega;
    if (t.size()==0) {
        assert(startn==0);
        const double Tmax = std::abs(Ereverse-Estart)*2;
        const int Nt = Tmax/dt;
#ifndef NDEBUG
        std::cout << "\tNt= "<<Nt<<std::endl;
#endif
        Itot.resize(Nt,0);
        t.resize(Nt);
        for (int i=0; i<Nt; i++) {
            t[i] = i*dt;
        }
    } else {
#ifndef NDEBUG
        std::cout << "\thave "<<t.size()<<" samples from "<<t[0]<<" to "<<t[t.size()-1]<<std::endl;
#endif
        Itot.resize(t.size(),0);
    }
    

    double phase_adjust = 0;
    if (startn != 0) {
        phase_adjust = std::abs(Ereverse-Estart)*omega;
    }
    Efun E(Estart,Ereverse,dE,omega,phase+phase_adjust,dt);

    vector a(Nx+1);
    vector b(Nx+1);
    vector c(Nx+1);
    for (int i=1; i<Nx; i++) {
        const double hstar = h[i]*h[i-1]*(h[i]+h[i-1]);
        a[i] = 2*h[i]/hstar;
        b[i] = 1/dt + 2*(h[i]+h[i-1])/hstar;
        c[i] = 2*h[i-1]/hstar;
    }
    vector d(Nx+1);
    vector e(Nx+1);
    vector f(Nx+1);
    e[Nx] = 0;
    f[Nx] = 1;

    for (int i=Nx-1; i >= 1; i--) {
        e[i]=a[i]/(b[i]-c[i]*e[i+1]);
    }
    
    vector U(Nx+1,1);
    

    double Itot0,Itot1,Itot_neg1;
    double t1 = 0;
    const double Itot_bound = std::max(10*Cdl*dE*omega/Nt,1.0);
    Itot_neg1 = Cdl*dE*omega;
    Itot0 = Itot_neg1;
    Itot1 = Itot_neg1;
    int n_out = startn;
    for (int dummy = 0; dummy < t.size(); dummy++) {
        while (t1 < t[n_out]) {
            Itot_neg1 = Itot0;
            Itot0 = Itot1;
            for (int i=1; i<Nx; i++) {
                d[i] = U[i]/dt;
            }
            for (int i=Nx-1; i>=1; i--) {
                f[i]=(d[i]+f[i+1]*c[i])/(b[i]-c[i]*e[i+1]);
            }

            const BCfun bc(h0,Cdl,f[1],e[1],E(t1+dt),E(t1),Ru,alpha,E0,dt,Itot0,k0);

            boost::uintmax_t max_it = max_iterations;
            Itot1 = boost::math::tools::newton_raphson_iterate(bc, Itot0,Itot0-Itot_bound,Itot0+Itot_bound, digits_accuracy, max_it);
            //Itot1 = boost::math::tools::bisect(bc,Itot0-1.1,Itot0+1.1,tol,max_it).first;
            if (max_it == max_iterations) throw std::runtime_error("non-linear solve for Itot[n+1] failed, max number of iterations reached");

            //std::cout << "residual is "<<bc.residual(Itot1)<<std::endl;
            //std::cout << "max_it "<<max_it<<std::endl;
            //std::cout << "residual gradient is "<<bc.residual_gradient(Itot1)<<std::endl;
            //std::cout << "If is "<<bc.If(Itot1)<<std::endl;
            //std::cout << "If2 is "<<bc.If2(Itot1)<<std::endl;
            U[0] = (f[1] - h0*bc.If(Itot1))/(1-e[1]);
            for (int i=1; i<Nx; i++) {
                U[i] = f[i] + e[i]*U[i-1];
            }
            t1 += dt;
        }
        // 2nd order interpolation
        const double x0 = t1;
        const double x1 = t1-dt;
        const double x2 = t1-2*dt;
        const double y0 = Itot1;
        const double y1 = Itot0;
        const double y2 = Itot_neg1;
        const double x = t[n_out];
        const double dt2 = dt*dt;
        //Itot[n_out] = -((x-x0)/dt) * y1 + ((x-x1)/dt) * y0;
        //Itot[n_out] = (Itot1-Itot0)*(t[n_out]-t1+dt)/dt + Itot0;
        Itot[n_out] = ((x-x0)*(x-x1) /(2*dt2)) * y2 + ((x-x0)*(x-x2) /(-dt2)) * y1 + ((x-x1)*(x-x2) /(2*dt2)) * y0;
        n_out++;
        if (n_out >= t.size()) {
            n_out = 0;
            E.set_phase(phase-phase_adjust);
        }
    }
}

}
