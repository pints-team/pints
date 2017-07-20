
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/tuple.hpp>
#include <math.h> 
#include "seq_electron_transfer3_explicit.hpp"
#include <iostream>
#include <exception>
#include <Eigen/Dense>

namespace pints {
template <unsigned int N>
struct seq_elec_fun {

    typedef Eigen::Matrix<double,2*N,2*N> matrix_type;
    typedef Eigen::Matrix<double,2*N,1> vector_type;

    double Cdl,CdlE,CdlE2,CdlE3;
    const double *E01,*E02;
    double Ru;
    const double *k01,*k02;
    const double *alpha1,*alpha2;
    double dt;
    const double *gamma;

    double dedt;
    vector_type u0,u1;
    double Cdlp;

    seq_elec_fun ( 
                    const double Cdl,
                    const double CdlE,
                    const double CdlE2,
                    const double CdlE3,
                    const double *E01,
                    const double *E02,
                    const double Ru,
                    const double *k01,
                    const double *k02,
                    const double *alpha1,
                    const double *alpha2,
                    const double dt,
                    const double *gamma

                    ) : 
        Cdl(Cdl),CdlE(CdlE),CdlE2(CdlE2),CdlE3(CdlE3),E01(E01),E02(E02),Ru(Ru),k01(k01),k02(k02),alpha1(alpha1),alpha2(alpha2),dt(dt),gamma(gamma) { }

    void init() {
        u0[0] = 1.0;
        for (int i=1; i<2*N; i++) {
            u0[i] = 0;
        }
    }

    double operator()(const double In0, const double E, const double dE) {
        update_concentrations(In0,E,dE);
        double In1 = Cdlp*(dE+Ru*In0/dt);
        In1 += gamma[0]*dedt;
        In1 /= (1.0 + Cdlp*Ru/dt);
        return In1;
    }


    void update_concentrations(const double In0, const double E, const double dE) {
        const double Ereduced = E - Ru*In0;
        const double Ereduced2 = pow(Ereduced,2);
        const double Ereduced3 = Ereduced*Ereduced2;

        // dudt = A*u + c
        // u1-u0 = dt*A*u1 + dt*c
        // u1-dt*A*u1 = u0 + dt*c 
        // (I-dt*A)*u1 = u0 + dt*c 
        // B*u1 = d 
        //
        // dedt = b'*u + c[-1]
        matrix_type A = matrix_type::Zero();
        vector_type b = vector_type::Zero();
        vector_type c = vector_type::Zero();
        double exp2o_old = 0;
        double exp2r_old = 0;
        double k02_old = 0;
        for (int i=0; i<N; i++) {
            const double expval1 = Ereduced - E01[i];
            const double expval2 = Ereduced - E02[i];

            const double exp1o = std::exp((1.0-alpha1[i])*expval1);
            const double exp1r = std::exp(-alpha1[i]*expval1);
            const double exp2o = std::exp((1.0-alpha2[i])*expval2);
            const double exp2r = std::exp(-alpha2[i]*expval2);

            if (i != 0) {
                A(2*i,2*i-1)  =  k02_old*exp2r_old;
                A(2*i,2*i)    = -k01[i]*exp1r-k02_old*exp2o_old;
                b(2*i)       += -k01[i]*exp1r;
            } else {
                A(2*i,2*i)    = -k01[i]*exp1r;
                b(2*i)       += -k01[i]*exp1r;
            } 
            exp2o_old = exp2o;
            exp2r_old = exp2r;
            k02_old = k02[i];
            A(2*i,2*i+1)  =  k01[i]*exp1o;
            b(2*i+1)     +=  k01[i]*exp1o;

            A(2*i+1,2*i)  =  k01[i]*exp1r;
            A(2*i+1,2*i+1)= -k02[i]*exp2r-k01[i]*exp1o;
            b(2*i+1)     += -k02[i]*exp2r;
            if (i != 2) {
                A(2*i+1,2*i+2)=  k02[i]*exp2o;
                b(2*i+2)     +=  k02[i]*exp2o;
            } else {
                c(2*i+1)      =  k02[i]*exp2o;
                for (int j = 0; j < 2*N; ++j) {
                    A(2*i+1,j) +=  -k02[i]*exp2o;
                    b(j)       +=  -k02[i]*exp2o;
                }
            }
        }
        //std::cout << "---------------"<<std::endl;
        //std::cout << A << std::endl;

        // integrate concentrations and calculate dudut
        u1 = (matrix_type::Identity()-dt*A).colPivHouseholderQr().solve(u0 + dt*c);
        dedt = (b.transpose()*u1)(0) + c[5];
        u0 = u1;

        Cdlp = Cdl*(1.0 + CdlE*Ereduced + CdlE2*Ereduced2 + CdlE3*Ereduced3);
    }
};

void seq_electron_transfer3_explicit(map& params, vector& Itot, vector& t) {
    const size_t N = 3;
    double k01[N],k02[N],alpha1[N],alpha2[N],E01[N],E02[N],gamma[N];

    k01[0] = get(params,std::string("k01"),35.0);
    k02[0] = get(params,std::string("k02"),65.0);
    E01[0] = get(params,std::string("E01"),0.25);
    E02[0] = get(params,std::string("E02"),-0.25);
    gamma[0] = get(params,std::string("gamma"),1.0);
    alpha1[0] = get(params,std::string("alpha1"),0.5);
    alpha2[0] = get(params,std::string("alpha2"),0.5);
    if (N>1) {
        k01[1] = get(params,std::string("k11"),35.0);
        k02[1] = get(params,std::string("k12"),65.0);
        alpha1[1] = get(params,std::string("alpha11"),0.5);
        alpha2[1] = get(params,std::string("alpha12"),0.5);
        E01[1] = get(params,std::string("E11"),0.25);
        E02[1] = get(params,std::string("E12"),-0.25);
        gamma[1] = get(params,std::string("gamma1"),1.0);
        if (N>2) {
            k01[2] = get(params,std::string("k21"),35.0);
            k02[2] = get(params,std::string("k22"),65.0);
            alpha1[2] = get(params,std::string("alpha21"),0.5);
            alpha2[2] = get(params,std::string("alpha22"),0.5);
            E01[2] = get(params,std::string("E21"),0.25);
            E02[2] = get(params,std::string("E22"),-0.25);
            gamma[2] = get(params,std::string("gamma2"),1.0);
        }
    }

    const double Ru = get(params,std::string("Ru"),0.001);
    const double Cdl = get(params,std::string("Cdl"),0.0037);
    const double CdlE = get(params,std::string("CdlE"),0.0);
    const double CdlE2 = get(params,std::string("CdlE2"),0.0);
    const double CdlE3 = get(params,std::string("CdlE3"),0.0);
    const double Estart = get(params,std::string("Estart"),-10.0);
    const double Ereverse = get(params,std::string("Ereverse"),10.0);
    const int Nt = get(params,std::string("Nt"),600.0);

    const double pi = boost::math::constants::pi<double>();
    const double omega = get(params,std::string("omega"),2*pi);
    const double phase = get(params,std::string("phase"),0.0);
    const double dE = get(params,std::string("dE"),0.1);
    const double reverse = 0;
    
    const int digits_accuracy = std::numeric_limits<double>::digits*0.5;
    const double max_iterations = N*100;

#ifndef NDEBUG
    std::cout << "Running seq_electron_transfer with parameters:"<<std::endl;
    for (int i=0; i<N; ++i) {
        std::cout << "\tk01 = "<<k01[i]<<std::endl;
        std::cout << "\tk02 = "<<k02[i]<<std::endl;
        std::cout << "\talpha1 = "<<alpha1[i]<<std::endl;
        std::cout << "\talpha2 = "<<alpha2[i]<<std::endl;
        std::cout << "\tE01 = "<<E01[i]<<std::endl;
        std::cout << "\tE02 = "<<E02[i]<<std::endl;
        std::cout << "\tgamma = "<<gamma[i]<<std::endl;
    }
    std::cout << "\tRu = "<<Ru<<std::endl;
    std::cout << "\tCdl = "<<Cdl<<std::endl;
    std::cout << "\tCdlE = "<<CdlE<<std::endl;
    std::cout << "\tCdlE2 = "<<CdlE2<<std::endl;
    std::cout << "\tCdlE3 = "<<CdlE3<<std::endl;
    std::cout << "\tEstart = "<<Estart<<std::endl;
    std::cout << "\tEreverse = "<<Ereverse<<std::endl;
    std::cout << "\tomega = "<<omega<<std::endl;
    std::cout << "\tphase = "<<phase<<std::endl;
    std::cout << "\tdE= "<<dE<<std::endl;
    std::cout << "\tNt= "<<Nt<<std::endl;
#endif
    //if (Ereverse < Estart) throw std::runtime_error("Ereverse must be greater than Estart");

    //set up temporal mesh
    const double dt = (1.0/Nt)*2*pi/omega;
    if (t.size()==0) {
        const double Tmax = std::abs(Ereverse-Estart)*2;
        const int Nt = Tmax/dt;
        std::cout << "\tNt= "<<Nt<<std::endl;
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
    

    Efun Eeq(Estart,Ereverse,dE,omega,phase,dt);

    double Itot0,Itot1;
    double t1 = 0;
    const double E = Eeq(t1);
    const double Cdlp = Cdl*(1.0 + CdlE*E + CdlE2*pow(E,2)+ CdlE3*pow(E,2));
    const double Itot_bound = std::max(10*Cdlp*dE*omega/Nt,1.0);

    Itot0 = Cdlp*Eeq.ddt(t1+0.5*dt);
    Itot1 = Itot0;
    seq_elec_fun<N> bc(Cdl,CdlE,CdlE2,CdlE3,E01,E02,Ru,k01,k02,alpha1,alpha2,dt,gamma);
    bc.init();

    for (int n_out = 0; n_out < t.size(); n_out++) {
        while (t1 < t[n_out]) {
            Itot0 = Itot1;
            const double E = Eeq(t1+dt);
            const double dE = Eeq.ddt(t1+0.5*dt);
            Itot1 = bc(Itot0,E,dE);
            t1 += dt;
        }

        //std::cout << "-----------------" << std::endl;
        //std::cout << bc.dudt << std::endl;
        Itot[n_out] = (Itot1-Itot0)*(t[n_out]-t1+dt)/dt + Itot0;
    }
}
}

