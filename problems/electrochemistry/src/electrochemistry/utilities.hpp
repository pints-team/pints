#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <cmath>
#include <map>
#include <iostream>

typedef std::vector<double> vector; 
typedef std::map<std::string,double> map; 

template <typename K, typename V>
V get(const  std::map <K,V> & m, const K & key, const V & defval ) {
    typename std::map<K,V>::const_iterator it = m.find( key );
    if ( it == m.end() ) {
        return defval;
    } else {
        return it->second;
    }
}

struct Efun {
    Efun() {};
    Efun(const Efun& other):
        Estart(other.Estart),Ereverse(other.Ereverse),
        dE(other.dE),omega(other.omega),phase(other.phase),dt(other.dt),
        treverse(other.treverse),direction(other.direction),
        dt_data(other.dt_data),Edata(other.Edata)
    {};
    Efun(const vector* Edata, const double dt_data, const double dE, const double omega, const double phase, const double dt):Edata(Edata),dt_data(dt_data),dE(dE),omega(omega),phase(phase),dt(dt) {};
    Efun(const double Estart, const double Ereverse, const double dE, const double omega, const double phase, const double dt):
        Estart(Estart),Ereverse(Ereverse),dE(dE),omega(omega),phase(phase),dt(dt),
        treverse(std::abs(Estart-Ereverse)),direction(Ereverse>Estart?1:-1),Edata(NULL) {
        };
    double operator[](const int n) const {
        const double t = n*dt;
        double Edc;
        if (Edata == NULL) {
            if (t<treverse) {
                Edc = Estart + direction*t;
            } else {
                Edc = Ereverse - direction*(t-treverse);
            }
        } else {
            const double x = t/dt_data;
            const unsigned int x0 = floor(x);
            const unsigned int x1 = ceil(x);
            const double y0 = (*Edata)[x0];
            const double y1 = (*Edata)[x1];
            Edc = ((x-x0)/dt_data) * y1 + ((x-x1)/dt_data) * y0;
        }

        return Edc + dE*std::sin(omega*t+phase);
    }
    double operator()(const double t) const {
        double Edc;
        if (Edata == NULL) {
            if (t<treverse) {
                Edc = Estart + direction*t;
            } else {
                Edc = Ereverse - direction*(t-treverse);
            }
        } else {
            const double x = t/dt_data;
            const unsigned int x0 = floor(x);
            const unsigned int x1 = ceil(x);
            const double y0 = (*Edata)[x0];
            const double y1 = (*Edata)[x1];
            Edc = (x-x0) * y1 + (x1-x) * y0;
        }
        return Edc + dE*std::sin(omega*t+phase);
    }
    double dc(const double t) const {
        double Edc;
        if (Edata == NULL) {
            if (t<treverse) {
                Edc = Estart + direction*t;
            } else {
                Edc = Ereverse - direction*(t-treverse);
            }
        } else {
            const double x = t/dt_data;
            const unsigned int x0 = floor(x);
            const unsigned int x1 = ceil(x);
            const double y0 = (*Edata)[x0];
            const double y1 = (*Edata)[x1];
            Edc = (x-x0) * y1 + (x1-x) * y0;
        }
        return Edc;
    }
    double ddt(const double t) const {
        double dEdcdt;
        if (Edata == NULL) {
            if (t>treverse) {
                dEdcdt = -direction;
            } else {
                dEdcdt = direction;
            }
        } else {
            const double x = t/dt_data;
            const unsigned int x0 = floor(x);
            const unsigned int x1 = ceil(x);
            const double y0 = (*Edata)[x0];
            const double y1 = (*Edata)[x1];
            dEdcdt = (y1-y0)/dt_data;
        }
        return dEdcdt + omega*dE*std::cos(omega*t+phase);
    }
    double ddt2(const double t) const {
        return -std::pow(omega,2)*dE*std::sin(omega*t+phase);
    }
    void set_phase(const double new_phase) { phase = new_phase; }
    double Estart,Ereverse,dE,omega,dt,treverse;
    double phase;
    int direction;
    const vector* Edata;
    double dt_data;
};


#endif
