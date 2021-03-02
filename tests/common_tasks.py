from loreleai.language.commons import c_pred, c_const,c_type,c_var, Atom
from loreleai.reasoning.lp.prolog import SWIProlog

"""
The description of these tasks is due to 
(Quinlan, 1990): Learning Logical definitions from Relations
"""


def family_relationships():
    """
    Family tree: https://i imgur com/43607jf png (Quinlan,1990)
    """

    wife = c_pred("wife",2)
    husband = c_pred("husband",2)
    mother = c_pred("mother",2)
    father = c_pred("father",2)
    daughter = c_pred("daughter",2)
    son = c_pred("son",2)
    sister = c_pred("sister",2)
    brother = c_pred("brother",2)
    aunt = c_pred("aunt",2)
    uncle = c_pred("uncle",2)
    niece = c_pred("niece",2)
    nephew = c_pred("nephew",2)

    christopher = c_const("christopher")
    penelope = c_const("penelope")
    margaret = c_const("margaret")
    arthur = c_const("arthur")
    victoria = c_const("victoria")
    james = c_const("james")
    colin = c_const("colin")
    charlotte = c_const("charlotte")
    andrew = c_const("andrew")
    christine = c_const("christine")
    jennifer = c_const("jennifer")
    charles = c_const("charles")
    roberto = c_const("roberto")
    maria = c_const("maria")
    gina = c_const("gina")
    emilio = c_const("emilio")
    lucia = c_const("lucia")
    marco = c_const("marco")
    alfonso = c_const("alfonso")
    sophia = c_const("sophia")
    pierro = c_const("pierro")
    francesca = c_const("francesca")
    angela = c_const("angela")
    tomaso = c_const("tomaso")

    bk = [
        wife(christopher,penelope),
        wife(andrew,christine),
        wife(charles,jennifer),
        wife(james,victoria),
        wife(arthur,margaret),
        wife(roberto,maria),
        wife(pierro,francesca),
        wife(emilio,gina),
        wife(marco,lucia),
        wife(tomaso,angela),

        husband(penelope,christopher),
        husband(christine,andrew),
        husband(jennifer,charles),
        husband(victoria,james),
        husband(margaret,arthur),
        husband(maria,roberto),
        husband(francesca,pierro),
        husband(gina,emilio),
        husband(lucia,marco),
        husband(angela,tomaso),

        mother(penelope,arthur),
        mother(penelope,victoria),
        mother(victoria,colin),
        mother(victoria,charlotte),
        mother(christine,jennifer),
        mother(christine,james),
        mother(maria,emilio),
        mother(maria,lucia),
        mother(lucia,alfonso),
        mother(lucia,sophia),
        mother(francesca,angela),
        mother(francesca,marco),

        father(christopher,arthur),
        father(christopher,victoria),
        father(james,colin),
        father(james,charlotte),
        father(andrew,james),
        father(andrew,jennifer),
        father(roberto,emilio),
        father(roberto,lucia),
        father(marco,alfonso),
        father(marco,sophia),
        father(pierro,marco),
        father(pierro,angela),

        daughter(victoria,christopher),
        daughter(victoria,penelope),
        daughter(charlotte,james),
        daughter(charlotte,victoria),
        daughter(jennifer,christine),
        daughter(jennifer,andrew),
        daughter(lucia,maria),
        daughter(lucia,roberto),
        daughter(sophia,marco),
        daughter(sophia,lucia),
        daughter(angela,francesca),
        daughter(angela,pierro),

        son(arthur,christopher),
        son(arthur,penelope),
        son(james,andrew),
        son(james,christine),
        son(colin,victoria),
        son(colin,charlotte),
        son(emilio,roberto),
        son(emilio,maria),
        son(marco,pierro),
        son(marco,francesca),
        son(alfonso,lucia),
        son(alfonso,marco),

        sister(charlotte,colin),
        sister(victoria,arthur),
        sister(jennifer,james),
        sister(lucia,emilio),
        sister(sophia,alfonso),
        sister(angela,marco),

        brother(colin,charlotte),
        brother(arthur,victoria),
        brother(james,jennifer),
        brother(emilio,lucia),
        brother(alfonso,sophia),
        brother(marco,angela),

        aunt(margaret,colin),
        aunt(margaret,charlotte),
        aunt(jennifer,colin),
        aunt(jennifer,charlotte),
        aunt(gina,alfonso),
        aunt(gina,sophia),
        aunt(angela,sophia),
        aunt(angela,alfonso),

        uncle(arthur,colin),
        uncle(arthur,charlotte),
        uncle(charles,colin),
        uncle(charles,charlotte),
        uncle(emilio,alfonso),
        uncle(emilio,sophia),
        uncle(tomaso,alfonso),
        uncle(tomaso,sophia),

        niece(charlotte,arthur),
        niece(charlotte,charles),
        niece(sophia,emilio),
        niece(sophia,tomaso),
        niece(charlotte,margaret),
        niece(charlotte,jennifer),
        niece(sophia,angela),
        niece(sophia,gina),

        nephew(colin,margaret),
        nephew(colin,jennifer),
        nephew(alfonso,gina),
        nephew(alfonso,angela),
        nephew(colin,arthur),
        nephew(colin,charles),
        nephew(alfonso,emilio),
        nephew(alfonso,tomaso),
    ]

    return bk


    

def michalski_trains():
    """
    Description of the classic eastbound/westbound trains problem (Michalski 1980)
    """
    car = c_type("car")
    train = c_type("train")
    shape = c_type("shape")
    num = c_type("num")

    car_shape = c_pred("car_shape",2,domains=(car,shape))
    short = c_pred("short",1,domains=(car))
    closed = c_pred("closed",1,domains=(car))
    long = c_pred("long",1,domains=(car))
    open_car = c_pred("open_car",1,domains=(car))
    load = c_pred("load",3,domains=(car,shape,num))
    wheels = c_pred("wheels",2,domains=(car,shape))
    has_car = c_pred("has_car",2,domains=(train,car))
    double = c_pred("double",1,domains=(car))
    jagged = c_pred("jagged",1,domains=(car))
    
    car_11 = c_const("car_11",domain=car) 
    car_12 = c_const("car_12",domain=car) 
    car_13 = c_const("car_13",domain=car) 
    car_14 = c_const("car_14",domain=car)
    car_21 = c_const("car_21",domain=car) 
    car_22 = c_const("car_22",domain=car) 
    car_23 = c_const("car_23",domain=car)
    car_31 = c_const("car_31",domain=car)
    car_32 = c_const("car_32",domain=car)
    car_33 = c_const("car_33",domain=car)
    car_41 = c_const("car_41",domain=car)
    car_42 = c_const("car_42",domain=car)
    car_43 = c_const("car_43",domain=car)
    car_44 = c_const("car_44",domain=car)
    car_51 = c_const("car_51",domain=car)
    car_52 = c_const("car_52",domain=car)
    car_53 = c_const("car_53",domain=car)
    car_61 = c_const("car_61",domain=car)
    car_62 = c_const("car_62",domain=car)
    car_71 = c_const("car_71",domain=car)
    car_72 = c_const("car_72",domain=car)
    car_73 = c_const("car_73",domain=car)
    car_81 = c_const("car_81",domain=car)
    car_82 = c_const("car_82",domain=car)
    car_91 = c_const("car_91",domain=car)
    car_92 = c_const("car_92",domain=car)
    car_93 = c_const("car_93",domain=car)
    car_94 = c_const("car_94",domain=car)
    car_101 = c_const("car_101",domain=car)
    car_102 = c_const("car_102",domain=car)

    east1 = c_const("east1",domain=train)  
    east2 = c_const("east2",domain=train)  
    east3 = c_const("east3",domain=train)  
    east4 = c_const("east4",domain=train)  
    east5 = c_const("east5",domain=train)
    west6 = c_const("west6",domain=train)  
    west7 = c_const("west7",domain=train)  
    west8 = c_const("west8",domain=train)  
    west9 = c_const("west9",domain=train)  
    west10 = c_const("west10",domain=train)

    elipse = c_const("elipse",domain=shape)  
    hexagon = c_const("hexagon",domain=shape)  
    rectangle = c_const("rectangle",domain=shape)  
    u_shaped = c_const("u_shaped",domain=shape)
    triangle = c_const("triangle",domain=shape) 
    circle = c_const("circle",domain=shape) 
    nil = c_const("nil",domain=shape)

    n0 = c_const("n0",domain=num)
    n1 = c_const("n1",domain=num)
    n2 = c_const("n2",domain=num)
    n3 = c_const("n3",domain=num)

    #eastbound train 1
    bk = [
        short(car_12),
        closed(car_12),
        long(car_11),
        long(car_13),
        short(car_14),
        open_car(car_11),
        open_car(car_13),
        open_car(car_14),
        car_shape(car_11,rectangle),
        car_shape(car_12,rectangle),
        car_shape(car_13,rectangle),
        car_shape(car_14,rectangle),
        load(car_11,rectangle,n3),
        load(car_12,triangle,n1),
        load(car_13,hexagon,n1),
        load(car_14,circle,n1),
        wheels(car_11,n2),
        wheels(car_12,n2),
        wheels(car_13,n3),
        wheels(car_14,n2),
        has_car(east1,car_11),
        has_car(east1,car_12),
        has_car(east1,car_13),
        has_car(east1,car_14),

        #eastbound train 
        has_car(east2,car_21),
        has_car(east2,car_22),
        has_car(east2,car_23),
        short(car_21),
        short(car_22),
        short(car_23),
        car_shape(car_21,u_shaped),
        car_shape(car_22,u_shaped),
        car_shape(car_23,rectangle),
        open_car(car_21),
        open_car(car_22),
        closed(car_23),
        load(car_21,triangle,n1),
        load(car_22,rectangle,n1),
        load(car_23,circle,n2),
        wheels(car_21,n2),
        wheels(car_22,n2),
        wheels(car_23,n2),

        #eastbound train
        has_car(east3,car_31),
        has_car(east3,car_32),
        has_car(east3,car_33),
        short(car_31),
        short(car_32),
        long(car_33),
        car_shape(car_31,rectangle),
        car_shape(car_32,hexagon),
        car_shape(car_33,rectangle),
        open_car(car_31),
        closed(car_32),
        closed(car_33),
        load(car_31,circle,n1),
        load(car_32,triangle,n1),
        load(car_33,triangle,n1),
        wheels(car_31,n2),
        wheels(car_32,n2),
        wheels(car_33,n3),

        #eastbound train
        has_car(east4,car_41),
        has_car(east4,car_42),
        has_car(east4,car_43),
        has_car(east4,car_44),
        short(car_41),
        short(car_42),
        short(car_43),
        short(car_44),
        car_shape(car_41,u_shaped),
        car_shape(car_42,rectangle),
        car_shape(car_43,elipse),
        car_shape(car_44,rectangle),
        double(car_42),
        open_car(car_41),
        open_car(car_42),
        closed(car_43),
        open_car(car_44),
        load(car_41,triangle,n1),
        load(car_42,triangle,n1),
        load(car_43,rectangle,n1),
        load(car_44,rectangle,n1),
        wheels(car_41,n2),
        wheels(car_42,n2),
        wheels(car_43,n2),
        wheels(car_44,n2),

        #eastbound train
        has_car(east5,car_51),
        has_car(east5,car_52),
        has_car(east5,car_53),
        short(car_51),
        short(car_52),
        short(car_53),
        car_shape(car_51,rectangle),
        car_shape(car_52,rectangle),
        car_shape(car_53,rectangle),
        double(car_51),
        open_car(car_51),
        closed(car_52),
        closed(car_53),
        load(car_51,triangle,n1),
        load(car_52,rectangle,n1),
        load(car_53,circle,n1),
        wheels(car_51,n2),
        wheels(car_52,n3),
        wheels(car_53,n2),

        #westbound train
        has_car(west6,car_61),
        has_car(west6,car_62),
        long(car_61),
        short(car_62),
        car_shape(car_61,rectangle),
        car_shape(car_62,rectangle),
        closed(car_61),
        open_car(car_62),
        load(car_61,circle,n3),
        load(car_62,triangle,n1),
        wheels(car_61,n2),
        wheels(car_62,n2),

        #westbound train
        has_car(west7,car_71),
        has_car(west7,car_72),
        has_car(west7,car_73),
        short(car_71),
        short(car_72),
        long(car_73),
        car_shape(car_71,rectangle),
        car_shape(car_72,u_shaped),
        car_shape(car_73,rectangle),
        double(car_71),
        open_car(car_71),
        open_car(car_72),
        jagged(car_73),
        load(car_71,circle,n1),
        load(car_72,triangle,n1),
        load(car_73,nil,n0),
        wheels(car_71,n2),
        wheels(car_72,n2),
        wheels(car_73,n2),

        #westbound train
        has_car(west8,car_81),
        has_car(west8,car_82),
        long(car_81),
        short(car_82),
        car_shape(car_81,rectangle),
        car_shape(car_82,u_shaped),
        closed(car_81),
        open_car(car_82),
        load(car_81,rectangle,n1),
        load(car_82,circle,n1),
        wheels(car_81,n3),
        wheels(car_82,n2),

        #westbound train
        has_car(west9,car_91),
        has_car(west9,car_92),
        has_car(west9,car_93),
        has_car(west9,car_94),
        short(car_91),
        long(car_92),
        short(car_93),
        short(car_94),
        car_shape(car_91,u_shaped),
        car_shape(car_92,rectangle),
        car_shape(car_93,rectangle),
        car_shape(car_94,u_shaped),
        open_car(car_91),
        jagged(car_92),
        open_car(car_93),
        open_car(car_94),
        load(car_91,circle,n1),
        load(car_92,rectangle,n1),
        load(car_93,rectangle,n1),
        load(car_93,circle,n1),
        wheels(car_91,n2),
        wheels(car_92,n2),
        wheels(car_93,n2),
        wheels(car_94,n2),

        # westbound train 1
        has_car(west10,car_101),
        has_car(west10,car_102),
        short(car_101),
        long(car_102),
        car_shape(car_101,u_shaped),
        car_shape(car_102,rectangle),
        open_car(car_101),
        open_car(car_102),
        load(car_101,rectangle,n1),
        load(car_102,rectangle,n2),
        wheels(car_101,n2),
        wheels(car_102,n2),
    ]
    return bk

if __name__ == "__main__":

    bk = michalski_trains()
    pl = SWIProlog()

    for fact in bk:
        pl.assert_fact(fact)

    x = c_var("X")
    y = c_var("Y")

    t = c_const("west10")
    has_car = c_pred("has_car",2)

    sols = pl.query(has_car(t,y))

    for sol in sols:
        print(f"has_car({t},{sol[y]})")




 




